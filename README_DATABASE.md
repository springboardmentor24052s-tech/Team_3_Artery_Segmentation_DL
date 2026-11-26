# Artery Segmentation Database Additions (Milestones 2 and 3)

## Included
- `app/models_milestone2_3.py`: additive SQLAlchemy models for segmentation, vessel analysis, stenosis candidate extraction, classification, localization, model evaluation, artifacts and reports.
- `scripts/init_db.py`: registers the existing and new models and creates missing tables.
- `.env.example`: local PostgreSQL configuration template.

## Setup
1. Create a project-specific virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env`.
3. Set `DATABASE_URL` to your local PostgreSQL database:
   `postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/artery_stenosis_db`
4. From the repository root, run:
   `python -m scripts.init_db`

## Important safety notes
- Never commit `.env` or real patient data.
- Back up the database before initialization.
- SQLAlchemy `create_all()` creates missing tables only; it does not migrate or alter existing tables.
- These additions use `ml_`-prefixed names for shared analysis tables to reduce collisions with legacy tables. Confirm the actual database schema and agree on table names with the team before production integration.
- Large masks, scans, overlays and model files belong in object storage. Store only their URI/path and metadata in PostgreSQL.
- Candidate coordinates and contour/centerline points are represented as JSON metadata; use a versioned coordinate convention agreed with the imaging/AI team.
- Model outputs are not clinical diagnoses. Keep review status and human-review workflows in place.
