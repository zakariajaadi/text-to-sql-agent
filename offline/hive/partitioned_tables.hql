-- Test tables reproducing the three read modes of the company data lake.
--   date_technique DATE      : ingestion date, partition column (Full, Delta)
--   `time`         TIMESTAMP : ingestion timestamp, version column for CDC
-- Backticks around `time`: TIME may be a reserved word (Hive 3, Impala).
USE chinook;

SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;

-- ── Full → Last ─────────────────────────────────────────────────────────────
-- One full customer snapshot per ingestion day. Only the latest partition is valid.
-- Day 1: 57 customers, day 2: 58, day 3: 59 with customer 1 moved to Rio de Janeiro.
DROP TABLE IF EXISTS customer_snapshot;
CREATE TABLE customer_snapshot (
  customerid INT, firstname STRING, lastname STRING, company STRING,
  address STRING, city STRING, state STRING, country STRING, postalcode STRING,
  phone STRING, fax STRING, email STRING, supportrepid INT,
  `time` TIMESTAMP
)
PARTITIONED BY (date_technique DATE)
STORED AS PARQUET;

INSERT OVERWRITE TABLE customer_snapshot PARTITION (date_technique = '2026-09-25')
SELECT customerid, firstname, lastname, company, address, city, state, country,
       postalcode, phone, fax, email, supportrepid,
       CAST('2026-09-25 00:00:00' AS TIMESTAMP)
FROM customer WHERE customerid <= 57;

INSERT OVERWRITE TABLE customer_snapshot PARTITION (date_technique = '2026-09-26')
SELECT customerid, firstname, lastname, company, address, city, state, country,
       postalcode, phone, fax, email, supportrepid,
       CAST('2026-09-26 00:00:00' AS TIMESTAMP)
FROM customer WHERE customerid <= 58;

INSERT OVERWRITE TABLE customer_snapshot PARTITION (date_technique = '2026-09-27')
SELECT customerid, firstname, lastname, company, address,
       CASE WHEN customerid = 1 THEN 'Rio de Janeiro' ELSE city END,
       state, country, postalcode, phone, fax, email, supportrepid,
       CAST('2026-09-27 00:00:00' AS TIMESTAMP)
FROM customer;

-- ── Delta/Append → All ──────────────────────────────────────────────────────
-- Invoices ingested monthly: each invoice lands in the partition of the month after it.
-- Every partition is valid; date_technique is NOT the business date (invoicedate is).
DROP TABLE IF EXISTS invoice_delta;
CREATE TABLE invoice_delta (
  invoiceid INT, customerid INT, invoicedate TIMESTAMP, billingaddress STRING,
  billingcity STRING, billingstate STRING, billingcountry STRING,
  billingpostalcode STRING, total DECIMAL(10,2),
  `time` TIMESTAMP
)
PARTITIONED BY (date_technique DATE)
STORED AS PARQUET;

-- Dynamic partition: the partition column must come last in the SELECT
INSERT OVERWRITE TABLE invoice_delta PARTITION (date_technique)
SELECT invoiceid, customerid, invoicedate, billingaddress, billingcity, billingstate,
       billingcountry, billingpostalcode, total,
       CAST(date_format(add_months(invoicedate, 1), 'yyyy-MM-01 00:00:00') AS TIMESTAMP),
       CAST(date_format(add_months(invoicedate, 1), 'yyyy-MM-01') AS DATE)
FROM invoice;

-- ── CDC → Live ──────────────────────────────────────────────────────────────
-- Updated rows only, no partition. Several versions per customerid: keep max(`time`).
-- v1: all 59 customers, v2: customers 1-5 changed email, v3: customer 1 moved again.
DROP TABLE IF EXISTS customer_cdc;
CREATE TABLE customer_cdc (
  customerid INT, firstname STRING, lastname STRING, city STRING, country STRING,
  email STRING, `time` TIMESTAMP, date_technique DATE
)
STORED AS PARQUET;

INSERT OVERWRITE TABLE customer_cdc
SELECT customerid, firstname, lastname, city, country, email,
       CAST('2026-09-01 00:00:00' AS TIMESTAMP), CAST('2026-09-01' AS DATE)
FROM customer
UNION ALL
SELECT customerid, firstname, lastname, city, country, concat('new.', email),
       CAST('2026-09-15 00:00:00' AS TIMESTAMP), CAST('2026-09-15' AS DATE)
FROM customer WHERE customerid <= 5
UNION ALL
SELECT customerid, firstname, lastname, 'Rio de Janeiro', country, concat('new.', email),
       CAST('2026-09-20 00:00:00' AS TIMESTAMP), CAST('2026-09-20' AS DATE)
FROM customer WHERE customerid = 1;