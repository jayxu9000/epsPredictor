drop database if exists epsPredictor;
create database epsPredictor;
use epsPredictor;

CREATE TABLE eps_analysis (
  id SERIAL PRIMARY KEY,
  company_symbol VARCHAR(20) NOT NULL,
  date DATE NOT NULL,
  reported_eps NUMERIC(10,4),
  estimated_eps NUMERIC(10,4),
  net_income BIGINT,
  sales DOUBLE,
  average_shares DOUBLE,
  income_from_continuing_operations DOUBLE,
  predicted_eps DOUBLE,
  difference DOUBLE
);