from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
import config

try:
    
    engine = create_engine(config.SQLALCHEMY_DATABASE_URI)

    
    with engine.connect() as connection:
        print("✅ Conexión exitosa.")

        
        inspector = inspect(engine)
        tablas = inspector.get_table_names()

        print("📋 Tablas encontradas en la base de datos:")
        for tabla in tablas:
            print(f"  - {tabla}")

except SQLAlchemyError as e:
    print("❌ Error al consultar:", e)
