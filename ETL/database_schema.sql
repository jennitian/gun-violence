-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.


CREATE TABLE "suspects" (
    "participant_index" int   NOT NULL,
    "incident_id" int,
    "participant_gender" varchar,
    "participant_age" float,
    "participant_age_group" varchar,
    "participant_status" varchar,
    "participant_type" varchar,
    CONSTRAINT "pk_suspects" PRIMARY KEY (
        "participant_index"
     )
);

CREATE TABLE "guns" (
    "gun_index" int   NOT NULL,
    "incident_id" int,
    "gun_type" varchar,
    "gun_stolen" varchar,
    "n_guns_involved" float,
    CONSTRAINT "pk_guns" PRIMARY KEY (
        "gun_index"
     )
);

CREATE TABLE "incidents" (
    "incident_id" int   NOT NULL,
    "date" date,
    "state" varchar,
    "latitude" float,
    "longitude" float,
    "n_killed" float,
    "n_injured" float,
    "incident_characteristics" varchar,
    "notes" varchar,
    "congressional_district" float,
    "state_house_district" float,
    "state_senate_district" float,
    CONSTRAINT "pk_incidents" PRIMARY KEY (
        "incident_id"
     )
);

ALTER TABLE "suspects" ADD CONSTRAINT "fk_suspects_incident_id" FOREIGN KEY("incident_id")
REFERENCES "incidents" ("incident_id");

ALTER TABLE "guns" ADD CONSTRAINT "fk_guns_incident_id" FOREIGN KEY("incident_id")
REFERENCES "incidents" ("incident_id");

CREATE TABLE "guns_ml_transformed" (
    "incident_id" int   NOT NULL,
    "n_guns_involved" float,
    "not_stolen" int,
    "stolen" int,
    "assault_rifle" int,
    "handgun" int,
    "rifle" int,
    "shotgun" int
);

CREATE TABLE "suspects_ml_transformed" (
    "incident_id" int   NOT NULL
    "participant_age" float,
    "female" int,
    "Adult_18+" int,
    "Child_0-11" int,
    "Teen_12-17" int,
    "status_Arrested" int,
    "status_Injured" int,
    "status_Injured,Arrested" int,
    "status_Killed" int,
    "status_Unharmed,Arrested" int
);
