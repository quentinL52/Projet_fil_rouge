from cv_parser import parse_cv
from postgres_storage import insert_into_db

pdf_path = r'data\CV_test.pdf'
result_df = parse_cv(pdf_path)

if not result_df.empty:
    insert_into_db(result_df)
else:
    print("Aucune donnée à insérer.")
