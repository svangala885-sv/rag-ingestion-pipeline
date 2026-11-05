# Cloud Cost Guidelines

## Cluster policies
Interactive clusters shut down after 30 minutes of inactivity. Scheduled jobs run on job clusters sized for the workload rather than on shared interactive clusters.

## Storage optimization
Large tables are partitioned by date and compacted weekly to avoid many small files. Tables not queried in 12 months move to a lower-cost storage tier.

## Budgets and alerts
Each team has a monthly cloud budget. An alert goes to the team lead at 80 percent of budget and to the platform team at 100 percent.
