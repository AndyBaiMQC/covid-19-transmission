# CGEM

# Data processing
To get Montreal seir values:
```
python model/process_mtl_infection.py
```

# Modelling
To fit model to real infection data:
```
python model/synthetic_fit.py
```
To run model without any NPIs:
```
python model/synthetic_exp.py
```
To run model with one NPI:
```
python model/npi_exp.py
```
To run model with one NPI and reopening:
```
python model/open_up_with_reset.py
```
To run model with all NPIs:
```
python model/contact_network_exp.py
```