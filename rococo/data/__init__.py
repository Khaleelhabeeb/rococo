"""data module"""

from .base import DbAdapter
import logging

logger = logging.getLogger(__name__)
__all__ = ["DbAdapter"]


# Conditional imports - only import if dependencies are available
try:
    from .mongodb import MongoDBAdapter
    __all__.append("MongoDBAdapter")
except ImportError:
    logger.info("MongoDBAdapter not loaded - probably, missing dependencies")
    pass

try:
    from .mysql import MySqlAdapter
    __all__.append("MySqlAdapter")
except ImportError:
    logger.info("MySqlAdapter not loaded - probably, missing dependencies")
    pass

try:
    from .postgresql import PostgreSQLAdapter
    __all__.append("PostgreSQLAdapter")
except ImportError:
    logger.info("PostgreSQLAdapter not loaded - probably, missing dependencies")
    pass

try:
    from .dynamodb import DynamoDbAdapter, DynamoOperation
    __all__.extend(["DynamoDbAdapter", "DynamoOperation"])
except ImportError:
    logger.info("DynamoDbAdapter not loaded - probably, missing dependencies")
    pass

try:
    from .surrealdb import SurrealDbAdapter
    __all__.append("SurrealDbAdapter")
except ImportError:
    logger.info("SurrealDbAdapter not loaded - probably, missing dependencies")
    pass
