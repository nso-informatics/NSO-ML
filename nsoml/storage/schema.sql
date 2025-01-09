CREATE TABLE combination (
    id SERIAL PRIMARY KEY,
    model VARCHAR(255) NOT NULL,
    resampler VARCHAR(255) NOT NULL,
    cross_validator VARCHAR(255) NOT NULL,
    scorer VARCHAR(255) NOT NULL,
    UNIQUE (model, resampler, cross_validator, scorer)
);

CREATE TABLE record (
    id SERIAL PRIMARY KEY,
    combination_id INTEGER REFERENCES combination,
    tag VARCHAR(255) NOT NULL,
    fold INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    latest BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE record_data (
    id SERIAL PRIMARY KEY,
    record_id INTEGER REFERENCES record,
    result_file VARCHAR(1023) NOT NULL,
    UNIQUE (record_id)
);

CREATE TABLE model_data (
    id SERIAL PRIMARY KEY,
    record_id INTEGER REFERENCES record,
    model_file VARCHAR(1023) NOT NULL,
    UNIQUE (record_id)
);

create TABLE analysis (
    id SERIAL PRIMARY KEY,
    record_id INTEGER REFERENCES record,
    f1 FLOAT NOT NULL DEFAULT -1.0,
    fp_fn FLOAT NOT NULL DEFAULT -1.0,
    fnr FLOAT NOT NULL DEFAULT -1.0,
    fpr FLOAT NOT NULL DEFAULT -1.0,
    tpr FLOAT NOT NULL DEFAULT -1.0,
    tnr FLOAT NOT NULL DEFAULT -1.0,
    accuracy FLOAT NOT NULL DEFAULT -1.0,
    sensitivity FLOAT NOT NULL DEFAULT -1.0,
    specificity FLOAT NOT NULL DEFAULT -1.0,
    precision_score FLOAT NOT NULL DEFAULT -1.0,
    recall_score FLOAT NOT NULL DEFAULT -1.0,
    ppv FLOAT NOT NULL DEFAULT -1.0,
    npv FLOAT NOT NULL DEFAULT -1.0,
    actual_positives INTEGER NOT NULL DEFAULT -1,
    actual_negatives INTEGER NOT NULL DEFAULT -1,
    false_positives INTEGER NOT NULL DEFAULT -1,
    false_negatives INTEGER NOT NULL DEFAULT -1,
    true_positives INTEGER NOT NULL DEFAULT -1,
    true_negatives INTEGER NOT NULL DEFAULT -1,
    total_predictions INTEGER NOT NULL DEFAULT -1,
    UNIQUE (record_id)
);

create TABLE optimal_feature_sets (
    id SERIAL PRIMARY KEY,
    combination_id INTEGER REFERENCES combination, 
    feature_set JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (combination_id)
);
