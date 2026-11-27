# Artery Segmentation Database Additions (Milestones 2 and 3)

For the full project architecture, user flows, model boundaries, API routes,
local setup, and current limitations, see [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md).

## Included
- `app/models_milestone2_3.py`: additive SQLAlchemy models for segmentation, vessel analysis, stenosis candidate extraction, classification, localization, model evaluation, artifacts and reports.
- `scripts/init_db.py`: registers the existing and new models and creates missing tables.
- `.env.example`: local PostgreSQL configuration template.

## Setup
1. Create a project-specific virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env`.
3. Set `DATABASE_URL` to your local PostgreSQL database:
   `postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/artery_stenosis_db`
4. Set `JWT_SECRET_KEY` to a unique random value.
5. From the repository root, start the database-integrated API with:
   `uvicorn app.main:app --reload`

The canonical application is `app.main:app`. `backend.main:app` is a separate
experimental image-processing application and is not the database-integrated
API; do not run both applications behind the same API base URL.

## Database-backed API
- Authentication: `POST /auth/register`, `POST /auth/login`, `GET /auth/me`.
- Patient records: `POST /patients`, `GET /patients`, `GET /patients/{id}`,
  and `PATCH /patients/{id}`.
- Cases: `POST /cases`, `GET /cases`, `GET /cases/{id}`, and
  `GET /patients/{id}/cases`.
- Images: upload with `POST /images` using `case_id` and `file` form fields;
  list with `GET /images?case_id={id}`; retrieve metadata/content with
  `GET /images/{id}` and `GET /images/{id}/content`.
- Image processing: `POST /images/{id}/preprocess` followed by
  `POST /images/{id}/predict`. Successful predictions are recorded in the
  existing `analysis_runs`, `segments`, `stenosis_findings`, `artifacts`, and
  `reports` tables and can be retrieved from
  `GET /cases/{id}/analysis-runs` or `GET /analysis-runs/{id}`.
- Baseline U-Net artery segmentation: upload with `POST /images` and then call
  `POST /images/{id}/segment`. The selected round-2 checkpoint is a separate
  pipeline from stenosis detection; mask, probability map, overlay, input
  preview, and model metadata use the existing analysis-run, model-version,
  segment, artifact, and report tables. Retrieve saved results through the same
  case analysis-run endpoints and authorized artifact route.
- Patient, case, image, and analysis endpoints require a bearer token. Users
  can access records they created; administrators can access all records.
- `POST /analyze-scan` and its `/upload-image` compatibility alias remain
  stateless for the existing UI. They validate and process an image but do not
  associate it with a case or persist an analysis run. The database-backed
  workflow is the authenticated `/images` API above. `model_type` is returned
  as the requested value, but the compatibility endpoint still uses the
  currently loaded team model.

## Important safety notes
- Never commit `.env` or real patient data.
- Back up the database before initialization.
- The database-integrated API does not run schema creation at startup.
  `scripts.init_db` uses SQLAlchemy `create_all()`, which creates missing tables
  only; it does not migrate or alter existing tables. Back up the database and
  review the live schema before running it.
- These additions use `ml_`-prefixed names for shared analysis tables to reduce collisions with legacy tables. Confirm the actual database schema and agree on table names with the team before production integration.
- Large masks, scans, overlays and model files belong in object storage. Store only their URI/path and metadata in PostgreSQL.
- Candidate coordinates and contour/centerline points are represented as JSON metadata; use a versioned coordinate convention agreed with the imaging/AI team.
- Model outputs are not clinical diagnoses. Keep review status and human-review workflows in place.
