# Open three json files
# Combine the list of dictionaries of input, output key values into one list

import json

with open("../../data/schemapile/ddl_filtered/schemapile_200_filtered-ddl-filtered.json") as f1, \
     open("../../data/schemapile/ddl_filtered/schemapile-pruned-sample432_end_filtered-ddl-filtered-sample-first450-with-nl.json") as f2, \
     open("../../data/schemapile/ddl_filtered/schemapile201_to_700_filtered-ddl-filtered.json") as f3:
    data1 = json.load(f1)
    data2 = json.load(f2)
    data3 = json.load(f3)

    combined_data = data1 + data2 + data3

    with open("../../data/schemapile/ddl_filtered/ddl-filtered-combined.json", "w") as f:
        json.dump(combined_data, f, indent=0)

    print(f"Combined dataset saved to ../CREATE TABLE Taco (
    id INT AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE Taco_Ingredients (
    taco_id BIGINT NOT NULL,
    ingredients_id VARCHAR(255) NOT NULL
);

CREATE TABLE Taco_Order (
    id INT AUTO_INCREMENT,
    user_id TINYINT NOT NULL,
    delivery_name VARCHAR(255) NOT NULL,
    delivery_street VARCHAR(255) NOT NULL,
    delivery_city VARCHAR(255) NOT NULL,
    delivery_state VARCHAR(255) NOT NULL,
    delivery_zip VARCHAR(255) NOT NULL,
    cc_number VARCHAR(255) NOT NULL,
    cc_expiration VARCHAR(255) NOT NULL,
    ccCVV VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE Taco_Order_Tacos (
    order_id BIGINT NOT NULL,
    tacos_id BIGINT NOT NULL
);
../data/schemapile/ddl_filtered/ddl-filtered-combined.json")