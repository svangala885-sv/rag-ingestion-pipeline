# Data Incident Response Runbook

## Severity levels
A Sev1 incident means incorrect data has reached an executive dashboard or an external report. A Sev2 incident means a critical pipeline has failed and data is late but not wrong. A Sev3 incident covers minor delays with no business impact.

## First response
For a Sev1, the on-call data engineer pauses the affected downstream jobs within 15 minutes, posts in the incident channel and notifies the data owner. Dashboards showing bad data get a banner that the numbers are under review.

## Post-incident review
Every Sev1 and Sev2 incident gets a blameless post-incident review within five business days, with action items tracked to completion.
