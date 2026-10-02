-- Separate database for MLflow's backend store; warehouse layer schemas.
-- Ingestion tables (raw.*, quarantine.*, meta.*) are created by retentioniq.ingestion.load.
CREATE DATABASE mlflow;

\connect retentioniq
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS quarantine;
CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS intermediate;
CREATE SCHEMA IF NOT EXISTS marts;
