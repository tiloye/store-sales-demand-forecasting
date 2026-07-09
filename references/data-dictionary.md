# Data Description
Schema definitions and feature descriptions for the Ecuadorian supermarket sales dataset

# Datasets

## train.csv

The training data, comprising time series of features used to train the model.

* **date** (date): The calendar date of the recorded data.
* **store_nbr** (integer): Unique identifier for the store at which the products are sold. Links to `stores.csv`.
* **family** (text): The category/type of product sold (e.g., AUTOMOTIVE, DAIRY, GROCERIES).
* **sales** (float): The target variable. Gives the total sales for a product family at a particular store on a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
* **onpromotion** (integer): The total number of items within a product family that were actively being promoted at a store on a given date.

## test.csv

The test data, containing the same features as the training data. Used to generate predictions for dates in the file. The dates in the test data are for the 15 days after the last date in the training data.

## sample_submission.csv

A sample submission file demonstrating the correct format required for evaluation.

* **id** (integer): Unique identifier for each row in the test set (test.csv).
* **sales** (float): The predicted target sales value for the corresponding row.

## stores.csv

Store metadata providing geographical and organizational details.

* **store_nbr** (integer): Unique identifier matching the `store_nbr` in train/test datasets.
* **city** (text): The city in Ecuador where the store is located.
* **state** (text): The state/province where the store is located.
* **type** (text): The structural category or format of the store.
* **cluster** (integer): A grouping of similar stores based on shared characteristics.

## oil.csv

Daily crude oil prices. Crucial for modeling economic health since Ecuador is an oil-dependent economy highly vulnerable to oil price shocks.

* **date** (date): The calendar date. Includes values spanning both train and test timeframes.
* **dcoilwtico** (float): Daily West Texas Intermediate (WTI) crude oil price. May contain missing values (e.g., weekends/holidays).

## holidays_events.csv

Local and national holidays, events, and related metadata.

* **date** (date): The calendar date of the holiday or event.
* **type** (text): The category of the day. Core types include:
* `Holiday`: A standard calendar holiday.
* `Transfer`: The date a holiday was actually celebrated if moved by the government.
* `Bridge`: Extra days added to extend a holiday into a long weekend.
* `Work Day`: A day not normally scheduled for work (e.g., a Saturday) meant to pay back a `Bridge` day.
* `Additional`: Days added around a regular calendar holiday (e.g., Christmas Eve).


* **locale** (text): The scope of the holiday (e.g., Local, Regional, National).
* **locale_name** (text): The specific name of the location affected (e.g., Ecuador, Guayaquil).
* **description** (text): The name of the specific holiday or event.
* **transferred** (boolean): Indicator of whether the holiday was officially moved. If `True`, the day acts like a normal day rather than a holiday. The actual celebration date will be listed on a separate row where `type` is `Transfer`.


# Additional Notes

* Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.

* A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.
