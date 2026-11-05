# Pipeline Engineering Standards

## Layers
Pipelines follow the Bronze, Silver and Gold medallion layers. Bronze stores data as received with ingestion metadata. Silver holds cleaned, deduplicated and conformed records. Gold holds business-level aggregates and dimensional models.

## Incremental loads
Pipelines load incrementally using a high-water mark column or change data capture. Full reloads require a written justification because of their compute cost.

## Code review and deployment
Every pipeline change needs one approving review and passing CI tests before merge. Deployments to production run through the CI/CD pipeline; manual changes in production are not allowed.
