from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import hashlib
import os
from pynamodb.models import Model
from pynamodb.connection import Connection
from pynamodb.transactions import TransactWrite
from pynamodb.attributes import Attribute, UnicodeAttribute, BooleanAttribute, NumberAttribute, JSONAttribute, ListAttribute
from pynamodb.exceptions import DoesNotExist, TransactWriteError
from rococo.data.base import DbAdapter
from rococo.models import BaseModel
from rococo.models.versioned_model import get_uuid_hex

__all__ = ["DynamoDbAdapter", "DynamoOperation", "DynamoPostCommitVersionMismatch"]

MAX_TRANSACTION_ITEMS = 100


class DynamoPostCommitVersionMismatch(RuntimeError):
    """Raised when the strongly-consistent read-back after a successful transaction
    commit returns a different version than expected.

    This indicates a subsequent writer overwrote the committed state between the
    transaction commit and the read-back. Callers can catch this specifically to
    distinguish post-commit races from other RuntimeErrors.
    """
    pass


@dataclass
class DynamoOperation:
    """Typed transaction operation for DynamoDB.

    For save/delete operations, `model` is a PynamoDB model instance. For
    audit_save operations, `model` is the source PynamoDB model class and
    `audit_model` is the destination audit model class.
    """
    action: str
    model: Union[Model, Type[Model]]
    condition: Any = None
    audit_model: Optional[Type[Model]] = None
    entity_id: Optional[str] = None
    expected_version: Optional[str] = None


class DynamoDbAdapter(DbAdapter):
    """DynamoDB adapter using PynamoDB with dynamic model generation.

    Thread-safety contract:
        The internal ``_pynamo_models`` and ``_connections`` caches use plain dicts
        with a check-then-set pattern that is **not** atomic under concurrent access.
        If the adapter instance is shared across threads (e.g., in a web server),
        two threads may both create a model class or connection for the same key,
        with one being silently discarded. This is benign (no data corruption or
        incorrect behaviour) but wastes one redundant connection setup. If strict
        single-instantiation semantics are required, callers should synchronize
        access externally or use one adapter instance per thread.
    """

    def __init__(self):
        self._connections: Dict[str, Connection] = {}
        self._pynamo_models: Dict[Tuple[str, Type[BaseModel], bool, str], Type[Model]] = {}
        self._model_classes_by_table: Dict[str, Type[BaseModel]] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _map_type_to_attribute(self, field_type: Any, is_hash_key: bool = False, is_range_key: bool = False):
        kwargs = {
            'hash_key': is_hash_key,
            'range_key': is_range_key,
            'null': True
        }
        
        if field_type == bool:
            return BooleanAttribute(**kwargs)
        elif field_type == int or field_type == float:
            return NumberAttribute(**kwargs)
        elif field_type == dict:
            return JSONAttribute(**kwargs)
        elif field_type == list:
            return ListAttribute(**kwargs)
        # Default to UnicodeAttribute for str and others
        return UnicodeAttribute(**kwargs)

    def _cache_digest(
        self,
        purpose: str,
        region: Optional[str],
        host: Optional[str],
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str]
    ) -> str:
        """Return an opaque cache key.

        Session tokens are intentionally omitted because temporary credentials
        are not cached. The purpose prefix prevents model-cache and connection-
        cache keys from sharing the same digest namespace.
        """
        key_material = "\x1f".join(
            value or "" for value in (
                purpose,
                region,
                host,
                aws_access_key_id,
                aws_secret_access_key
            )
        )
        return hashlib.sha256(key_material.encode("utf-8")).hexdigest()

    def _model_cache_digest(
        self,
        region: Optional[str],
        host: Optional[str],
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str]
    ) -> str:
        return self._cache_digest(
            "model",
            region,
            host,
            aws_access_key_id,
            aws_secret_access_key
        )

    def _resolve_model_cls(self, table_name: str, model_cls: Type[BaseModel] = None) -> Type[BaseModel]:
        if model_cls is not None:
            self._model_classes_by_table[table_name] = model_cls
            return model_cls

        cached_model_cls = self._model_classes_by_table.get(table_name)
        if cached_model_cls is not None:
            return cached_model_cls

        raise ValueError("model_cls is required for DynamoDB operations until the table model has been registered")

    def _generate_pynamo_model(self, table_name: str, model_cls: Type[BaseModel], is_audit: bool = False) -> Type[Model]:
        """Dynamically generate a PynamoDB Model class from a Rococo BaseModel or VersionedModel."""
        model_cls = self._resolve_model_cls(table_name, model_cls)
        
        # 1. Define Meta
        class Meta:
            table_name_val = table_name
            region = os.getenv('AWS_REGION', 'us-east-1')
            host = os.getenv('DYNAMODB_ENDPOINT_URL') or None
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID') or None
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY') or None
            aws_session_token = os.getenv('AWS_SESSION_TOKEN') or None

        cache_digest = self._model_cache_digest(
            Meta.region,
            Meta.host,
            Meta.aws_access_key_id,
            Meta.aws_secret_access_key
        )
        cache_key = (table_name, model_cls, is_audit, cache_digest)
        if not Meta.aws_session_token and cache_key in self._pynamo_models:
            return self._pynamo_models[cache_key]
        
        attrs = {
            'Meta': type('Meta', (), {
                'table_name': Meta.table_name_val,
                'region': Meta.region,
                'host': Meta.host,
                'aws_access_key_id': Meta.aws_access_key_id,
                'aws_secret_access_key': Meta.aws_secret_access_key,
                'aws_session_token': Meta.aws_session_token
            })
        }

        # 2. Map fields
        # If it's an audit table, we might want a different key structure
        # Standard Rococo Audit: entity_id (Hash), version (Range)
        
        if is_audit:
            attrs['entity_id'] = UnicodeAttribute(hash_key=True)
            attrs['version'] = UnicodeAttribute(range_key=True)
        else:
            # Standard Table: entity_id (Hash)
            attrs['entity_id'] = UnicodeAttribute(hash_key=True)

        # Add other fields from dataclass
        if hasattr(model_cls, '__dataclass_fields__'):
            for field_name, field_def in model_cls.__dataclass_fields__.items():
                if field_name == 'entity_id':
                    continue
                if is_audit and field_name == 'version':
                    continue
                
                attrs[field_name] = self._map_type_to_attribute(field_def.type)

        # 3. Create class
        class_name = f"Pynamo{model_cls.__name__}{'Audit' if is_audit else ''}"
        pynamo_model = type(class_name, (Model,), attrs)
        if not Meta.aws_session_token:
            self._pynamo_models[cache_key] = pynamo_model
        return pynamo_model

    def _build_connection(
        self,
        region: str,
        host: Optional[str],
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        aws_session_token: Optional[str]
    ) -> Connection:
        return Connection(
            region=region,
            host=host,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )

    def _connection_cache_key(
        self,
        region: str,
        host: Optional[str],
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str]
    ) -> str:
        return self._cache_digest(
            "connection",
            region,
            host,
            aws_access_key_id,
            aws_secret_access_key
        )

    def _get_connection(self, model: Union[Model, Type[Model]]) -> Connection:
        meta = getattr(model, "Meta", None)
        region = getattr(meta, "region", None) or os.getenv("AWS_REGION", "us-east-1")
        host = getattr(meta, "host", None) or os.getenv("DYNAMODB_ENDPOINT_URL") or None
        aws_access_key_id = getattr(meta, "aws_access_key_id", None) or os.getenv("AWS_ACCESS_KEY_ID") or None
        aws_secret_access_key = getattr(meta, "aws_secret_access_key", None) or os.getenv("AWS_SECRET_ACCESS_KEY") or None
        aws_session_token = getattr(meta, "aws_session_token", None) or os.getenv("AWS_SESSION_TOKEN") or None

        if aws_session_token:
            return self._build_connection(
                region,
                host,
                aws_access_key_id,
                aws_secret_access_key,
                aws_session_token
            )

        key = self._connection_cache_key(region, host, aws_access_key_id, aws_secret_access_key)
        if key not in self._connections:
            self._connections[key] = self._build_connection(
                region,
                host,
                aws_access_key_id,
                aws_secret_access_key,
                None
            )
        return self._connections[key]

    def _validate_transaction_operations(self, ops: List[Any]) -> List["DynamoOperation"]:
        for op in ops:
            if not isinstance(op, DynamoOperation):
                raise RuntimeError("DynamoDbAdapter.run_transaction expects only DynamoOperation objects")
            if op.action not in {"save", "delete", "audit_save"}:
                raise ValueError(f"Unknown DynamoOperation action: '{op.action}'")
        return ops

    def _table_name_for(self, model: Union[Model, Type[Model]]) -> Optional[str]:
        return getattr(getattr(model, "Meta", None), "table_name", None)

    def _entity_id_for(self, model: Model) -> Optional[str]:
        attribute_values = getattr(model, "attribute_values", {}) or {}
        return attribute_values.get("entity_id") or getattr(model, "entity_id", None)

    def _operation_source_key(self, op: DynamoOperation) -> Optional[Tuple[str, str]]:
        if op.action == "audit_save":
            if op.entity_id is None:
                return None
            table_name = self._table_name_for(op.model)
            if table_name is None:
                return None
            return (table_name, op.entity_id)

        if op.action in {"save", "delete"} and isinstance(op.model, Model):
            entity_id = self._entity_id_for(op.model)
            if entity_id is None:
                return None
            table_name = self._table_name_for(op.model)
            if table_name is None:
                return None
            return (table_name, entity_id)

        return None

    def _source_write_guards(self, ops: List[DynamoOperation]) -> Dict[Tuple[str, str], str]:
        guards: Dict[Tuple[str, str], str] = {}
        for op in ops:
            if op.action not in {"save", "delete"} or op.condition is None or op.expected_version is None:
                continue

            source_key = self._operation_source_key(op)
            if source_key is not None:
                guards[source_key] = op.expected_version

        return guards

    def _is_source_guarded_by_write(
        self,
        op: DynamoOperation,
        source_write_guards: Dict[Tuple[str, str], str]
    ) -> bool:
        source_key = self._operation_source_key(op)
        if source_key is None:
            return False
        return source_write_guards.get(source_key) == op.expected_version

    def _estimate_transaction_item_count(self, ops: List[DynamoOperation]) -> int:
        source_write_guards = self._source_write_guards(ops)
        item_count = 0
        for op in ops:
            item_count += 1
            if op.action == "audit_save" and not self._is_source_guarded_by_write(op, source_write_guards):
                item_count += 1
        return item_count

    def _validate_transaction_item_count(self, ops: List[DynamoOperation]) -> None:
        item_count = self._estimate_transaction_item_count(ops)
        if item_count > MAX_TRANSACTION_ITEMS:
            raise ValueError(
                f"DynamoDB transactions support at most {MAX_TRANSACTION_ITEMS} items; got {item_count}"
            )

    def _version_condition(self, model_cls: Type[Model], expected_version: str):
        version_attr = getattr(model_cls, "version", None)
        if not isinstance(version_attr, Attribute):
            table_name = self._table_name_for(model_cls) or "<unknown>"
            raise RuntimeError(
                f"Versioned DynamoDB operation for table '{table_name}' requires a PynamoDB 'version' attribute"
            )
        return version_attr == expected_version

    def _resolve_audit_operation(self, op: DynamoOperation) -> DynamoOperation:
        """Resolve a deferred audit_save into a concrete save operation.

        TOCTOU note:
            The audit snapshot is read via a strongly-consistent GetItem *before*
            the TransactWrite batch is committed. Between this read and the commit,
            a concurrent writer could change non-version fields on the source row.
            The version condition (ConditionCheck or same-row save condition) guards
            against a stale *version* being committed — if the version changes, the
            transaction aborts. However, non-version field changes that do not
            alter the version will slip through silently into the audit snapshot.

            This is an inherent limitation of DynamoDB's TransactWriteItems API,
            which does not support read-then-write within a single atomic action.
            Only version integrity is guaranteed; field-level consistency of the
            audit snapshot is best-effort.

        See: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/transaction-apis.html
        """
        if op.audit_model is None or op.entity_id is None:
            raise RuntimeError("Dynamo audit operation requires audit_model and entity_id")
        if op.expected_version is None or op.expected_version == "":
            raise RuntimeError("Dynamo audit operation requires a non-empty expected_version")

        # Best-effort snapshot read. The authoritative ACID boundary is the
        # later same-row save condition or audit-only ConditionCheck.
        try:
            item = op.model.get(op.entity_id, consistent_read=True)
        except DoesNotExist:
            raise RuntimeError(
                f"DynamoDB audit snapshot source entity_id={op.entity_id} does not exist"
            )

        audit_item = op.audit_model(**item.attribute_values)
        return DynamoOperation("save", audit_item, condition=op.condition)

    def _add_source_condition_check(self, transaction, op: DynamoOperation) -> None:
        transaction.condition_check(
            op.model,
            op.entity_id,
            condition=self._version_condition(op.model, op.expected_version)
        )

    def _apply_transaction_operation(self, transaction, op: DynamoOperation) -> None:
        if op.action == "save":
            transaction.save(op.model, condition=op.condition)
        elif op.action == "delete":
            transaction.delete(op.model, condition=op.condition)
        else:
            # Defensive unreachable path: run_transaction validates actions and
            # audit_save operations are resolved before dispatch.
            raise ValueError(f"Unknown DynamoOperation action: '{op.action}'")

    def run_transaction(self, operations_list: List[Any]):
        """Execute operations as a single DynamoDB ACID transaction.

        This adapter intentionally requires `DynamoOperation` entries and will
        raise if anything else is provided.
        """
        ops = [op for op in (operations_list or []) if op is not None]
        if not ops:
            return

        ops = self._validate_transaction_operations(ops)
        self._validate_transaction_item_count(ops)
        source_write_guards = self._source_write_guards(ops)

        # Derive connection from the first save/delete op (which always carries a
        # concrete model instance) rather than ops[0] which may be an audit_save
        # class with potentially different Meta region/host.
        connection_source = next(
            (op.model for op in ops if op.action in {"save", "delete"}),
            ops[0].model
        )
        connection = self._get_connection(connection_source)

        try:
            with TransactWrite(connection=connection) as transaction:
                for op in ops:
                    resolved_op = op
                    if op.action == "audit_save":
                        resolved_op = self._resolve_audit_operation(op)
                        if not self._is_source_guarded_by_write(op, source_write_guards):
                            self._add_source_condition_check(transaction, op)

                    self._apply_transaction_operation(transaction, resolved_op)
        except (TransactWriteError, ValueError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Transaction failed: {e}") from e

    def execute_query(self, sql: str, _vars: Dict[str, Any] = None) -> Any:
        raise NotImplementedError("execute_query is not supported for DynamoDB")

    def parse_db_response(self, response: Any) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if isinstance(response, list):
            return [item.attribute_values for item in response]
        if isinstance(response, Model):
            return response.attribute_values
        return response

    def get_one(self, table: str, conditions: Dict[str, Any], sort: List[Tuple[str, str]] = None, model_cls: Type[BaseModel] = None) -> Dict[str, Any]:
        model_cls = self._resolve_model_cls(table, model_cls)
            
        pynamo_model = self._generate_pynamo_model(table, model_cls)
        try:
            results = self._execute_query_or_scan(pynamo_model, conditions, limit=1)
            # results is an iterator
            for item in results:
                return item.attribute_values
            return None
        except Exception as e:
            raise RuntimeError(f"get_one failed: {e}")

    def get_many(self, table: str, conditions: Dict[str, Any] = None, sort: List[Tuple[str, str]] = None, limit: int = 100, model_cls: Type[BaseModel] = None) -> List[Dict[str, Any]]:
        model_cls = self._resolve_model_cls(table, model_cls)

        pynamo_model = self._generate_pynamo_model(table, model_cls)
        try:
            results = self._execute_query_or_scan(pynamo_model, conditions, limit=limit)
            return [item.attribute_values for item in results]
        except Exception as e:
            raise RuntimeError(f"get_many failed: {e}")

    def get_count(self, table: str, conditions: Dict[str, Any], options: Optional[Dict[str, Any]] = None, model_cls: Type[BaseModel] = None) -> int:
        model_cls = self._resolve_model_cls(table, model_cls)

        pynamo_model = self._generate_pynamo_model(table, model_cls)
        try:
            return self._execute_query_or_scan(pynamo_model, conditions, count_only=True)
        except Exception as e:
            raise RuntimeError(f"get_count failed: {e}")

    def get_move_entity_to_audit_table_query(
        self,
        table,
        entity_id,
        model_cls: Type[BaseModel] = None,
        expected_version: str = None
    ):
        """Return a deferred operation that snapshots the current item into audit.

        The source item is read inside run_transaction, not while building the
        operation. The main-table save operation must still carry the
        previous_version condition; that condition is the commit-time guard that
        aborts the whole transaction if another writer changes the source row
        after the audit snapshot read.
        """
        model_cls = self._resolve_model_cls(table, model_cls)

        pynamo_model = self._generate_pynamo_model(table, model_cls)
        audit_table_name = f"{table}_audit"
        pynamo_audit_model = self._generate_pynamo_model(audit_table_name, model_cls, is_audit=True)

        return DynamoOperation(
            "audit_save",
            pynamo_model,
            condition=pynamo_audit_model.version.does_not_exist(),
            audit_model=pynamo_audit_model,
            entity_id=entity_id,
            expected_version=expected_version
        )

    def move_entity_to_audit_table(self, table_name: str, entity_id: str, model_cls: Type[BaseModel] = None):
        model_cls = self._resolve_model_cls(table_name, model_cls)

        pynamo_model = self._generate_pynamo_model(table_name, model_cls)
        audit_table_name = f"{table_name}_audit"
        pynamo_audit_model = self._generate_pynamo_model(audit_table_name, model_cls, is_audit=True)

        try:
            item = pynamo_model.get(entity_id, consistent_read=True)
            audit_item = pynamo_audit_model(**item.attribute_values)
            audit_item.save()
        except DoesNotExist:
            pass
        except Exception as e:
            raise RuntimeError(f"move_entity_to_audit_table failed: {e}")

    def get_save_query(self, table: str, data: Dict[str, Any], model_cls: Type[BaseModel] = None):
        model_cls = self._resolve_model_cls(table, model_cls)

        pynamo_model = self._generate_pynamo_model(table, model_cls)
        item = pynamo_model(**data)

        # Optimistic locking / create condition:
        # - New records have previous_version == ZERO_UUID (set by prepare_for_save)
        # - Updated records have previous_version == the previous stored version
        previous_version = data.get("previous_version")
        if "version" in data and previous_version is None:
            raise RuntimeError(
                "Versioned DynamoDB save requires previous_version; for new records it must be the zero UUID sentinel"
            )
        if "version" in data and previous_version == "":
            raise RuntimeError("Versioned DynamoDB save requires non-empty previous_version")

        if previous_version is not None and previous_version != get_uuid_hex(0):
            # PynamoDB exposes condition-capable attributes on the model class.
            # Instance-level values are plain Python data and cannot build expressions.
            condition = self._version_condition(pynamo_model, previous_version)
            operation_expected_version = previous_version
        else:
            condition = pynamo_model.entity_id.does_not_exist()
            operation_expected_version = None

        return DynamoOperation(
            "save",
            item,
            condition=condition,
            entity_id=data.get("entity_id"),
            expected_version=operation_expected_version
        )

    def save(self, table: str, data: Dict[str, Any], model_cls: Type[BaseModel] = None) -> Union[Dict[str, Any], None]:
        """Unconditional PutItem save. NOT safe for versioned models.

        For versioned models, use get_save_query() + run_transaction() which
        applies optimistic locking conditions. This method performs a plain
        PutItem with no version guard and will silently overwrite concurrent
        changes.

        Raises:
            RuntimeError: If data contains a 'version' field, indicating a
                versioned model that should use the transactional save path.
        """
        if "version" in data:
            raise RuntimeError(
                "save() must not be used for versioned models — it bypasses optimistic locking. "
                "Use get_save_query() + run_transaction() instead."
            )
        model_cls = self._resolve_model_cls(table, model_cls)

        pynamo_model = self._generate_pynamo_model(table, model_cls)
        item = pynamo_model(**data)
        item.save()
        return item.attribute_values

    def get_by_id(
        self,
        table: str,
        entity_id: str,
        model_cls: Type[BaseModel] = None,
        consistent_read: bool = False
    ) -> Optional[Dict[str, Any]]:
        model_cls = self._resolve_model_cls(table, model_cls)

        pynamo_model = self._generate_pynamo_model(table, model_cls)
        try:
            item = pynamo_model.get(entity_id, consistent_read=consistent_read)
            return item.attribute_values
        except DoesNotExist:
            return None

    def upsert(self, table: str, data: Dict[str, Any], model_cls: Type[BaseModel] = None) -> Union[Dict[str, Any], None]:
        """
        Upsert (update or insert) an item in the specified DynamoDB table.

        This is a simple upsert operation for non-versioned models that replaces
        the entire item based on entity_id (hash key).

        Args:
            table (str): The name of the DynamoDB table.
            data (Dict[str, Any]): The item data (must include 'entity_id').
            model_cls (Type[BaseModel]): The model class for schema mapping.

        Returns:
            Dict[str, Any]: The upserted item's attribute values.

        Raises:
            ValueError: If model_cls is not provided.
            RuntimeError: If entity_id is missing or any DynamoDB operation fails.
        """
        model_cls = self._resolve_model_cls(table, model_cls)

        if 'entity_id' not in data:
            raise RuntimeError("upsert failed: 'entity_id' is required in data")

        try:
            pynamo_model = self._generate_pynamo_model(table, model_cls)
            item = pynamo_model(**data)
            item.save()  # PynamoDB's save() is already an upsert (put_item)
            return item.attribute_values
        except Exception as e:
            raise RuntimeError(f"upsert failed: {e}")

    def delete(self, table: str, data: Dict[str, Any], model_cls: Type[BaseModel] = None) -> bool:
        model_cls = self._resolve_model_cls(table, model_cls)

        pynamo_model = self._generate_pynamo_model(table, model_cls)
        entity_id = data.get('entity_id')
        if entity_id:
            try:
                item = pynamo_model.get(entity_id)
                item.active = False
                item.save()
                return True
            except DoesNotExist:
                return False
        return False

    def hard_delete(self, table: str, entity_id: str, model_cls: Type[BaseModel] = None) -> bool:
        """Permanently deletes a record from the specified table by entity_id."""
        model_cls = self._resolve_model_cls(table, model_cls)

        pynamo_model = self._generate_pynamo_model(table, model_cls)
        try:
            item = pynamo_model.get(entity_id)
            item.delete()
            return True
        except DoesNotExist:
            return False

    def _execute_query_or_scan(self, model_cls: Type[Model], conditions: Dict[str, Any], limit: int = None, count_only: bool = False):
        """
        Helper to determine whether to use Query or Scan based on conditions.
        """
        # Find hash key and range key using public API instead of _meta
        hash_key_name = None
        range_key_name = None
        
        for name, attr in model_cls.get_attributes().items():
            if getattr(attr, 'is_hash_key', False):
                hash_key_name = name
            if getattr(attr, 'is_range_key', False):
                range_key_name = name

        hash_key_val = conditions.get(hash_key_name) if conditions else None
        
        if hash_key_val is not None:
            # Query path: Hash key is present
            range_key_condition = None
            filter_condition = None
            
            for key, value in conditions.items():
                if key == hash_key_name:
                    continue
                
                attr = getattr(model_cls, key)
                cond = (attr == value)
                
                if key == range_key_name:
                    range_key_condition = cond
                else:
                    if filter_condition is None:
                        filter_condition = cond
                    else:
                        filter_condition = filter_condition & cond
            
            if count_only:
                return model_cls.count(hash_key_val, range_key_condition=range_key_condition, filter_condition=filter_condition)
            else:
                return model_cls.query(hash_key_val, range_key_condition=range_key_condition, filter_condition=filter_condition, limit=limit)
        else:
            # Scan path: Hash key is missing
            scan_condition = None
            if conditions:
                for key, value in conditions.items():
                    attr = getattr(model_cls, key)
                    cond = (attr == value)
                    if scan_condition is None:
                        scan_condition = cond
                    else:
                        scan_condition = scan_condition & cond
            
            if count_only:
                return model_cls.count(filter_condition=scan_condition)
            else:
                return model_cls.scan(scan_condition, limit=limit)
