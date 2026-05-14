import unittest
import os
from uuid import uuid4
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, ANY
from typing import Type
from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute, BooleanAttribute
from pynamodb.exceptions import DoesNotExist, TransactWriteError
from rococo.data.dynamodb import DynamoDbAdapter, DynamoOperation
from rococo.repositories.dynamodb.dynamodb_repository import DynamoDbRepository
from rococo.models.versioned_model import VersionedModel, get_uuid_hex
from rococo.messaging import MessageAdapter

# Dummy PynamoDB Models
class PersonModel(Model):
    class Meta:
        table_name = 'person'
        region = 'us-east-1'
    entity_id = UnicodeAttribute(hash_key=True)
    first_name = UnicodeAttribute(null=True)
    last_name = UnicodeAttribute(null=True)
    active = BooleanAttribute(default=True)
    version = UnicodeAttribute(null=True)
    previous_version = UnicodeAttribute(null=True)
    changed_by_id = UnicodeAttribute(null=True)
    changed_on = UnicodeAttribute(null=True)
    attribute_values = {} 

    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        self.attribute_values = kwargs

    def save(self):
        pass
    
    def delete(self):
        pass

class PersonAuditModel(Model):
    class Meta:
        table_name = 'person_audit'
        region = 'us-east-1'
    entity_id = UnicodeAttribute(hash_key=True)
    first_name = UnicodeAttribute(null=True)
    last_name = UnicodeAttribute(null=True)
    active = BooleanAttribute(default=True)
    version = UnicodeAttribute(null=True)
    previous_version = UnicodeAttribute(null=True)
    changed_by_id = UnicodeAttribute(null=True)
    changed_on = UnicodeAttribute(null=True)
    attribute_values = {}

    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        self.attribute_values = kwargs

    def save(self):
        pass

class NoVersionModel:
    class Meta:
        table_name = 'person'
        region = 'us-east-1'
    entity_id = UnicodeAttribute(hash_key=True)

    def __init__(self, **kwargs):
        self.attribute_values = kwargs

# Dummy Rococo Model
@dataclass
class Person(VersionedModel):
    first_name: str = None
    last_name: str = None

class TestDynamoDbRepository(unittest.TestCase):
    def setUp(self):
        os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
        self.adapter = DynamoDbAdapter()
        self.message_adapter = MagicMock(spec=MessageAdapter)
        self.repository = DynamoDbRepository(
            self.adapter,
            Person,
            self.message_adapter,
            'test_queue'
        )

        # Patch _generate_pynamo_model to return our dummy models
        self.patcher = patch.object(self.adapter, '_generate_pynamo_model', side_effect=self._mock_generate_model)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def _mock_generate_model(self, table_name, model_cls, is_audit=False):
        if is_audit:
            return PersonAuditModel
        return PersonModel

    def test_get_one_query(self):
        # Test get_one using query (hash key present)
        # Mock get_attributes for _execute_query_or_scan
        PersonModel.get_attributes = MagicMock()
        
        # Mock attributes to identify hash key
        mock_hash_attr = MagicMock()
        mock_hash_attr.is_hash_key = True
        mock_hash_attr.is_range_key = False
        
        PersonModel.get_attributes.return_value = {'entity_id': mock_hash_attr}
        
        entity_id = uuid4().hex

        with patch.object(PersonModel, 'query') as mock_query:
            mock_item = MagicMock()
            mock_item.attribute_values = {'entity_id': entity_id, 'first_name': 'John', 'active': True}
            mock_query.return_value = [mock_item]

            result = self.repository.get_one({'entity_id': entity_id})
            
            self.assertIsNotNone(result)
            self.assertEqual(result.entity_id, entity_id)
            self.assertEqual(result.first_name, 'John')
            mock_query.assert_called_once()

    def test_get_one_scan(self):
        # Test get_one using scan (hash key missing)
        # Mock get_attributes for _execute_query_or_scan
        PersonModel.get_attributes = MagicMock()
        
        # Mock attributes to identify hash key
        mock_hash_attr = MagicMock()
        mock_hash_attr.is_hash_key = True
        mock_hash_attr.is_range_key = False
        
        PersonModel.get_attributes.return_value = {'entity_id': mock_hash_attr}
        
        entity_id = uuid4().hex

        with patch.object(PersonModel, 'scan') as mock_scan:
            mock_item = MagicMock()
            mock_item.attribute_values = {'entity_id': entity_id, 'first_name': 'John', 'active': True}
            mock_scan.return_value = [mock_item]

            result = self.repository.get_one({'first_name': 'John'})
            
            self.assertIsNotNone(result)
            self.assertEqual(result.first_name, 'John')
            mock_scan.assert_called_once()

    def test_save_new(self):
        # Test saving a new record
        person = Person(first_name='Jane', last_name='Doe')

        with patch.object(self.adapter, 'get_by_id') as mock_get_by_id:
            mock_get_by_id.side_effect = lambda table, entity_id, model_cls, consistent_read: {
                'entity_id': entity_id,
                'first_name': 'Jane',
                'last_name': 'Persisted',
                'active': True,
                'version': person.version,
            }
            transact_patch = patch('rococo.data.dynamodb.TransactWrite')
            mock_transact_write = transact_patch.start()
            self.addCleanup(transact_patch.stop)
            mock_tx = MagicMock()
            mock_transact_write.return_value.__enter__.return_value = mock_tx

            saved_person = self.repository.save(person, send_message=True)

            self.assertEqual(saved_person.first_name, 'Jane')
            self.assertEqual(saved_person.last_name, 'Persisted')
            mock_get_by_id.assert_called_once()
            self.assertTrue(mock_get_by_id.call_args.kwargs["consistent_read"])
            mock_tx.save.assert_called()
            condition = mock_tx.save.call_args.kwargs["condition"]
            self.assertIn("attribute_not_exists", str(condition))
            self.assertIn("entity_id", str(condition))
            
            # Verify message was sent
            self.message_adapter.send_message.assert_called_with(
                'test_queue', 
                ANY  # The message body (JSON)
            )

    def test_save_existing_audit(self):
        # Test saving an existing record (should trigger audit)
        person = Person(first_name='Jane', last_name='Doe')
        entity_id = uuid4().hex
        old_version = uuid4().hex
        person.entity_id = entity_id
        person.version = old_version  # Set version so prepare_for_save preserves it in previous_version
        person.previous_version = old_version
        
        # Mock the 'get' call used by move_entity_to_audit_table
        with patch.object(PersonModel, 'get') as mock_get:
            mock_item = MagicMock()
            mock_item.attribute_values = {
                'entity_id': entity_id,
                'first_name': 'Jane',
                'active': True,
                'version': old_version,
            }
            mock_get.return_value = mock_item
            
            with patch.object(self.adapter, 'get_by_id') as mock_get_by_id:
                mock_get_by_id.side_effect = lambda table, entity_id, model_cls, consistent_read: {
                    'entity_id': entity_id,
                    'first_name': 'Jane',
                    'last_name': 'Doe',
                    'active': True,
                    'version': person.version,
                    'previous_version': old_version,
                }
                transact_patch = patch('rococo.data.dynamodb.TransactWrite')
                mock_transact_write = transact_patch.start()
                self.addCleanup(transact_patch.stop)
                mock_tx = MagicMock()
                mock_transact_write.return_value.__enter__.return_value = mock_tx

                self.repository.save(person, send_message=True)

                # Ensure we tried to fetch the old record to audit it
                mock_get.assert_called_with(entity_id, consistent_read=True)

                # Ensure we called transaction.save twice (audit + new)
                self.assertEqual(mock_tx.save.call_count, 2)
                audit_condition = mock_tx.save.call_args_list[0].kwargs["condition"]
                save_condition = mock_tx.save.call_args_list[1].kwargs["condition"]
                self.assertIn("attribute_not_exists", str(audit_condition))
                self.assertIn("version", str(audit_condition))
                self.assertIn(old_version, str(save_condition))
                mock_tx.condition_check.assert_not_called()
                mock_get_by_id.assert_called_once()

                # Verify message was sent
                self.message_adapter.send_message.assert_called()

    def test_delete(self):
        person = Person(first_name='Jane')
        person.entity_id = uuid4().hex

        with patch.object(self.adapter, 'get_by_id') as mock_get_by_id:
            mock_get_by_id.side_effect = lambda table, entity_id, model_cls, consistent_read: {
                'entity_id': entity_id,
                'first_name': 'Jane',
                'active': False,
                'version': person.version,
            }
            transact_patch = patch('rococo.data.dynamodb.TransactWrite')
            mock_transact_write = transact_patch.start()
            self.addCleanup(transact_patch.stop)
            mock_tx = MagicMock()
            mock_transact_write.return_value.__enter__.return_value = mock_tx

            self.repository.delete(person)
            self.assertFalse(person.active)
            mock_get_by_id.assert_called_once()
            mock_tx.save.assert_called()

    def test_get_move_entity_to_audit_table_query_is_deferred(self):
        entity_id = uuid4().hex
        old_version = uuid4().hex

        with patch.object(PersonModel, 'get') as mock_get:
            operation = self.adapter.get_move_entity_to_audit_table_query(
                'person',
                entity_id,
                model_cls=Person,
                expected_version=old_version
            )

        mock_get.assert_not_called()
        self.assertEqual(operation.action, "audit_save")
        self.assertEqual(operation.entity_id, entity_id)
        self.assertEqual(operation.expected_version, old_version)

    def test_audit_operation_defers_snapshot_version_mismatch_to_transaction_condition(self):
        entity_id = uuid4().hex
        old_version = uuid4().hex
        operation = self.adapter.get_move_entity_to_audit_table_query(
            'person',
            entity_id,
            model_cls=Person,
            expected_version=old_version
        )

        with patch.object(PersonModel, 'get') as mock_get:
            mock_item = MagicMock()
            mock_item.attribute_values = {
                'entity_id': entity_id,
                'version': uuid4().hex,
            }
            mock_get.return_value = mock_item

            with patch('rococo.data.dynamodb.TransactWrite') as mock_transact_write:
                mock_tx = MagicMock()
                mock_transact_write.return_value.__enter__.return_value = mock_tx

                self.adapter.run_transaction([operation])

        mock_tx.condition_check.assert_called_once()
        self.assertIn(old_version, str(mock_tx.condition_check.call_args.kwargs["condition"]))
        mock_tx.save.assert_called_once()

    def test_audit_operation_rejects_missing_source_item(self):
        entity_id = uuid4().hex
        operation = self.adapter.get_move_entity_to_audit_table_query(
            'person',
            entity_id,
            model_cls=Person,
            expected_version=uuid4().hex
        )

        with patch.object(PersonModel, 'get', side_effect=DoesNotExist()):
            with patch('rococo.data.dynamodb.TransactWrite') as mock_transact_write:
                mock_tx = MagicMock()
                mock_transact_write.return_value.__enter__.return_value = mock_tx

                with self.assertRaisesRegex(RuntimeError, "does not exist"):
                    self.adapter.run_transaction([operation])

        mock_tx.condition_check.assert_not_called()
        mock_tx.save.assert_not_called()

    def test_audit_operation_rejects_empty_expected_version(self):
        entity_id = uuid4().hex
        operation = self.adapter.get_move_entity_to_audit_table_query(
            'person',
            entity_id,
            model_cls=Person,
            expected_version=""
        )

        with patch.object(PersonModel, 'get') as mock_get:
            with patch('rococo.data.dynamodb.TransactWrite') as mock_transact_write:
                mock_tx = MagicMock()
                mock_transact_write.return_value.__enter__.return_value = mock_tx

                with self.assertRaisesRegex(RuntimeError, "non-empty expected_version"):
                    self.adapter.run_transaction([operation])

        mock_get.assert_not_called()
        mock_tx.save.assert_not_called()

    def test_audit_only_operation_adds_source_condition_check(self):
        entity_id = uuid4().hex
        old_version = uuid4().hex
        operation = self.adapter.get_move_entity_to_audit_table_query(
            'person',
            entity_id,
            model_cls=Person,
            expected_version=old_version
        )

        with patch.object(PersonModel, 'get') as mock_get:
            mock_item = MagicMock()
            mock_item.attribute_values = {
                'entity_id': entity_id,
                'first_name': 'Jane',
                'active': True,
                'version': old_version,
            }
            mock_get.return_value = mock_item

            with patch('rococo.data.dynamodb.TransactWrite') as mock_transact_write:
                mock_tx = MagicMock()
                mock_transact_write.return_value.__enter__.return_value = mock_tx

                self.adapter.run_transaction([operation])

        mock_tx.condition_check.assert_called_once()
        condition_check_call = mock_tx.condition_check.call_args
        self.assertIs(condition_check_call.args[0], PersonModel)
        self.assertEqual(condition_check_call.args[1], entity_id)
        self.assertIn(old_version, str(condition_check_call.kwargs["condition"]))
        mock_tx.save.assert_called_once()

    def test_get_save_query_uses_shared_zero_uuid_sentinel(self):
        data = {
            'entity_id': uuid4().hex,
            'previous_version': get_uuid_hex(0),
            'version': uuid4().hex,
            'active': True,
        }

        operation = self.adapter.get_save_query('person', data, model_cls=Person)

        self.assertIn("attribute_not_exists", str(operation.condition))
        self.assertIn("entity_id", str(operation.condition))

    def test_get_save_query_requires_version_attribute_for_updates(self):
        data = {
            'entity_id': uuid4().hex,
            'previous_version': uuid4().hex,
            'version': uuid4().hex,
        }

        with patch.object(self.adapter, '_generate_pynamo_model', return_value=NoVersionModel):
            with self.assertRaisesRegex(RuntimeError, "requires a PynamoDB 'version' attribute"):
                self.adapter.get_save_query('person', data, model_cls=Person)

    def test_get_save_query_rejects_missing_previous_version_for_versioned_data(self):
        data = {
            'entity_id': uuid4().hex,
            'version': uuid4().hex,
        }

        with self.assertRaisesRegex(RuntimeError, "requires previous_version"):
            self.adapter.get_save_query('person', data, model_cls=Person)

    def test_get_save_query_rejects_empty_previous_version_for_versioned_data(self):
        data = {
            'entity_id': uuid4().hex,
            'version': uuid4().hex,
            'previous_version': "",
        }

        with self.assertRaisesRegex(RuntimeError, "non-empty previous_version"):
            self.adapter.get_save_query('person', data, model_cls=Person)

    def test_run_transaction_reraises_transact_write_error(self):
        operation = DynamoOperation("save", PersonModel(entity_id=uuid4().hex))
        error = TransactWriteError("conditional check failed")

        with patch('rococo.data.dynamodb.TransactWrite') as mock_transact_write:
            mock_transact_write.return_value.__enter__.side_effect = error

            with self.assertRaises(TransactWriteError) as context:
                self.adapter.run_transaction([operation])

        self.assertIs(context.exception, error)

    def test_run_transaction_validates_operations_before_opening_transaction(self):
        valid_operation = DynamoOperation("save", PersonModel(entity_id=uuid4().hex))

        with patch('rococo.data.dynamodb.TransactWrite') as mock_transact_write:
            with self.assertRaisesRegex(RuntimeError, "expects only DynamoOperation objects"):
                self.adapter.run_transaction([valid_operation, object()])

        mock_transact_write.assert_not_called()

    def test_run_transaction_rejects_unknown_action_before_opening_transaction(self):
        operation = DynamoOperation("typo", PersonModel(entity_id=uuid4().hex))

        with patch('rococo.data.dynamodb.TransactWrite') as mock_transact_write:
            with self.assertRaisesRegex(ValueError, "Unknown DynamoOperation action"):
                self.adapter.run_transaction([operation])

        mock_transact_write.assert_not_called()

    def test_run_transaction_enforces_dynamodb_item_limit(self):
        operations = [
            DynamoOperation("save", PersonModel(entity_id=uuid4().hex))
            for _ in range(101)
        ]

        with patch('rococo.data.dynamodb.Connection') as mock_connection:
            with patch('rococo.data.dynamodb.TransactWrite') as mock_transact_write:
                with self.assertRaisesRegex(ValueError, "at most 100 items"):
                    self.adapter.run_transaction(operations)

        mock_connection.assert_not_called()
        mock_transact_write.assert_not_called()

    def test_run_transaction_reuses_cached_connection(self):
        adapter = DynamoDbAdapter()
        operation = DynamoOperation("save", PersonModel(entity_id=uuid4().hex))

        # Empty string simulates no temporary session token.
        with patch.dict(os.environ, {'AWS_SESSION_TOKEN': ''}):
            with patch('rococo.data.dynamodb.Connection') as mock_connection:
                with patch('rococo.data.dynamodb.TransactWrite'):
                    adapter.run_transaction([operation])
                    adapter.run_transaction([operation])

        self.assertEqual(mock_connection.call_count, 1)

    def test_connection_cache_key_does_not_expose_secret(self):
        adapter = DynamoDbAdapter()
        operation = DynamoOperation("save", PersonModel(entity_id=uuid4().hex))
        secret = "very-secret-value"

        with patch.dict(os.environ, {'AWS_SECRET_ACCESS_KEY': secret, 'AWS_SESSION_TOKEN': ''}):
            with patch('rococo.data.dynamodb.Connection'):
                with patch('rococo.data.dynamodb.TransactWrite'):
                    adapter.run_transaction([operation])

        self.assertNotIn(secret, repr(adapter._connections))

    def test_model_and_connection_cache_keys_use_separate_namespaces(self):
        adapter = DynamoDbAdapter()
        args = ("us-east-1", None, "testing", "secret")

        self.assertNotEqual(
            adapter._model_cache_digest(*args),
            adapter._connection_cache_key(*args)
        )

    def test_generated_pynamo_model_is_cached_without_session_token(self):
        adapter = DynamoDbAdapter()

        # Empty string simulates no temporary session token.
        with patch.dict(os.environ, {'AWS_SESSION_TOKEN': ''}):
            first_model = adapter._generate_pynamo_model('person', Person)
            second_model = adapter._generate_pynamo_model('person', Person)

        self.assertIs(first_model, second_model)

    def test_get_move_entity_to_audit_table_query_can_use_registered_model_cls(self):
        entity_id = uuid4().hex
        old_version = uuid4().hex
        self.adapter.get_save_query(
            'person',
            {
                'entity_id': entity_id,
                'version': uuid4().hex,
                'previous_version': get_uuid_hex(0),
            },
            model_cls=Person
        )

        operation = self.adapter.get_move_entity_to_audit_table_query(
            'person',
            entity_id,
            expected_version=old_version
        )

        self.assertEqual(operation.action, "audit_save")
        self.assertEqual(operation.expected_version, old_version)

    def test_read_committed_state_rejects_version_mismatch(self):
        entity_id = uuid4().hex
        expected_version = uuid4().hex

        with patch.object(self.adapter, 'get_by_id') as mock_get_by_id:
            mock_get_by_id.return_value = {
                'entity_id': entity_id,
                'version': uuid4().hex,
            }

            with self.assertRaisesRegex(RuntimeError, "write succeeded"):
                self.repository._read_committed_state(entity_id, expected_version=expected_version)

    def test_session_token_connections_are_not_cached(self):
        adapter = DynamoDbAdapter()
        operation = DynamoOperation("save", PersonModel(entity_id=uuid4().hex))

        with patch.dict(os.environ, {'AWS_SESSION_TOKEN': 'short-lived-token'}):
            with patch('rococo.data.dynamodb.Connection') as mock_connection:
                with patch('rococo.data.dynamodb.TransactWrite'):
                    adapter.run_transaction([operation])
                    adapter.run_transaction([operation])

        self.assertEqual(mock_connection.call_count, 2)
        self.assertEqual(adapter._connections, {})

    def test_dynamo_operation_is_publicly_exported(self):
        from rococo.data import DynamoOperation as ExportedDynamoOperation

        self.assertIs(ExportedDynamoOperation, DynamoOperation)


if __name__ == '__main__':
    unittest.main()
