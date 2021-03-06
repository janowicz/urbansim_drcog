# Opus/UrbanSim urban simulation software.
# Copyright (C) 2010-2011 University of California, Berkeley, 2005-2009 University of Washington
# See opus_core/LICENSE 

calibration_paris = {
    'xml_config' : '/home/atschirhar/opus/project_configs/paris_zone.xml',
    'scenario' : 'paris_zone_calibration2',
    'calib_datasets' : {'establishment_location_choice_model_coefficients': 'estimate'},
    'subset' : None,
    'subset_patterns' : {'establishment_location_choice_model_coefficients':['coefficient_name', '_celcm$']},
    'target_expression' : "zgpgroup.aggregate((establishment.employment)*(establishment.disappeared==0),intermediates=[building,zone,zgp])",
    'target_file' : '/workspace/opus/data/paris_zone/temp_data/zgpgroup_totemp00.csv',
    'skip_cache_cleanup': True,
    'optimizer' : 'bfgs',
}
    
calibration_bayarea_developer = {
    'xml_config' : '/workspace/opus/project_configs/bay_area_parcel_unit_price.xml',
    'scenario' : 'developer_calibration',
    'calib_datasets' : {'cost_shifter': ['residential_shifter', 'non_residential_shifter'], 'price_shifter': 'estimate'},
    'subset' : None,
    'subset_patterns' : None,
    'target_expression' : "devcalib_geography.aggregate((building.residential_units*(building.building_type_id<4)) + (building.non_residential_sqft*(building.building_type_id>3)))",
    'target_file' : '/workspace/opus/data/bay_area_parcel/calibration_targets/county_development2011.csv',
    'skip_cache_cleanup': True,
    'optimizer' : 'lbfgsb',
    'optimizer_kwargs': {'epsilon': 1e-3}
}

calibration_bayarea_hlcm = {
    'xml_config' : '/workspace/opus/project_configs/bay_area_parcel_unit_price.xml',
    'scenario' : 'hlcm_calibration',
    'calib_datasets' : {'submarket_household_location_choice_model_owner_coefficients': 'estimate','submarket_household_location_choice_model_renter_coefficients': 'estimate'},
    'subset' : None,
    'subset_patterns' : {'submarket_household_location_choice_model_owner_coefficients':['coefficient_name', '_calib$'],'submarket_household_location_choice_model_renter_coefficients':['coefficient_name', '_calib$']},
    'target_expression' : "county.aggregate(submarket.number_of_agents(household))",
    'target_file' : '/workspace/opus/data/bay_area_parcel/calibration_targets/county_hh2011.csv',
    'skip_cache_cleanup': False,
    'optimizer' : 'bfgs',
}

calibration_bayarea_elcm_priceeq = {
    'xml_config' : '/workspace/opus/project_configs/bay_area_parcel_unit_price_priceeq.xml',
    'scenario' : 'elcm_calibration_w_priceeq',
    'calib_datasets' : {'business_location_choice_model_coefficients': 'estimate'},
    'subset' : None,
    'subset_patterns' : {'business_location_choice_model_coefficients': ['coefficient_name', 'ln_avg_nonres_rent']},
    'target_expression' : "empcalib_group.aggregate(establishment.employees)",
    'target_file' : '/workspace/opus/data/bay_area_parcel/calibration_targets/county_employment_2011.csv',
    'skip_cache_cleanup': False,
    'optimizer' : 'bfgs'
}
