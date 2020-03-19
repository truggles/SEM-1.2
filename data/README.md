# Electricity and Gas Prices by Regions

## EIA data for U.S.

Data from https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_nus_w.htm

U.S. total and by state

I am using the "Gasoline - All Grades" category which appears to average over regular, mid, and premium fuel types.

Also using industrial electricity prices from: https://www.eia.gov/electricity/sales_revenue_price/
See table T5.c 	Industrial average monthly bill by Census Division, and State and file `eia_elec_industry_by_state_table5_c_2018.xlsx`

To align with SEDS (below) we use 2017 values from their archive: https://www.eia.gov/electricity/sales_revenue_price/archive/f8612017.zip (still Table 5.c)
`eia_elec_industry_by_state_table5_c_2017.xlsx`

EIA provides gasoline price estimates for all states, 1970-2017 in their State Energy Data System (SEDS).
 * https://www.eia.gov/tools/faqs/faq.php?id=26&t=10
 * https://www.eia.gov/state/seds/seds-data-complete.php?sid=US#CompleteDataFile
 * https://www.eia.gov/state/seds/seds-technical-notes-complete.php#Prices%20and%20Expenditures
See file here: `eia_SEDS_1970-2017.csv`

To get Btu / barrel use: https://www.eia.gov/totalenergy/data/monthly/pdf/sec12_4.pdf
5.053 MMBtu/barrel in 2017
And: 42 gallons / US barrel


## Global data

See file `Global_elec_and_gas_prices.csv`

with data from: https://www.globalpetrolprices.com/
Elec price is for businesses for month of June 2019
Gasoline price is for all users 09-Mar-2020
