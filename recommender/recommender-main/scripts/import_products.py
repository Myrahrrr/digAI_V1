import pandas as pd
from app import create_app
from models import db
from models.product import Product  # ajuste si ton modèle est ailleurs

CSV_PATH = "products.csv"

def main():
    app = create_app()
    with app.app_context():
        df = pd.read_csv(CSV_PATH)

        added = 0
        for _, r in df.iterrows():
            name = str(r["name"]).strip()

            exists = Product.query.filter_by(name=name).first()
            if exists:
                continue

            p = Product(
                name=name,
                x=float(r["x"]),
                y=float(r["y"]),
                type=str(r["type"]).strip(),
                fabric=str(r["fabric"]).strip(),
                pattern=str(r["pattern"]).strip(),
                size=str(r["size"]).strip(),
                color=str(r["color"]).strip(),
            )
            db.session.add(p)
            added += 1

        db.session.commit()
        print(f"import ok, added={added}, total={Product.query.count()}")

if __name__ == "__main__":
    main()
