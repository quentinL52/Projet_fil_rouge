import os
from sqlalchemy import create_engine

def get_engine():
    db_url = os.getenv("DB_URL", "postgresql+psycopg2://postgres:postgres@localhost:5433/projet_fil_rouge")
    return create_engine(db_url)

def read_system_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def chatbot_instructions_centered(file_path):
    import shutil
    with open(file_path, 'r', encoding='utf-8') as file:
        instructions = file.read()
    console_width = shutil.get_terminal_size().columns
    centered_lines = [line.center(console_width) for line in instructions.split('\n')]
    centered_text = "\n".join(centered_lines) 
    quoted_centered_text = f'{centered_text}'
    return quoted_centered_text