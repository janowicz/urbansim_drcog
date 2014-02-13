# Opus/UrbanSim urban simulation software.
# Copyright (C) 2010-2011 University of California, Berkeley, 2005-2009 University of Washington
# See opus_core/LICENSE 

from opus_core.model import Model
from opus_core.logger import logger
from opus_core.session_configuration import SessionConfiguration
import numpy as np, pandas as pd

class Urbansim2(Model):
    """Runs an UrbanSim2 scenario
    """
    model_name = "UrbanSim2"
    
    def __init__(self,scenario='Base Scenario'):
        self.scenario = scenario

    def run(self, name=None, export_buildings_to_urbancanvas=False, base_year=2010, forecast_year=None, fixed_seed=True, random_seed=1, export_indicators=True, indicator_output_directory='C:/opus/data/drcog2/runs', core_components_to_run=None, household_transition=None,household_relocation=None,employment_transition=None, elcm_configuration=None, developer_configuration=None, calibration_configuration=None, hh_targets=None, emp_targets=None):
        """Runs an UrbanSim2 scenario 
        """
        logger.log_status('Starting UrbanSim2 run.')
        
        import numpy as np, pandas as pd, os, statsmodels.api as sm
        import synthicity.urbansim.interaction as interaction
        from synthicity.utils import misc
        import dataset, copy, time, math
        np.random.seed(1)
        
        resunit_targets = np.array([.198,.205,.105,.032,.002,.165,.142,.014,.002,.099,.037])
        #hh_targets = np.array([.198,.205,.105,.032,.002,.165,.142,.014,.002,.099,.037])
        hh_targets = np.array([hh_targets['hh_8001_target'],hh_targets['hh_8005_target'],hh_targets['hh_8013_target'],hh_targets['hh_8014_target'],hh_targets['hh_8019_target'],hh_targets['hh_8031_target'],hh_targets['hh_8035_target'],hh_targets['hh_8039_target'],hh_targets['hh_8047_target'],hh_targets['hh_8059_target'],hh_targets['hh_8123_target']])
        nonres_targets = np.array([0.1511,0.2232,0.0737,0.0473,0.0001,0.2435,0.1094,0.0139,0.0005,0.1178,0.0197])
        #emp_targets = np.array([0.1511,0.2232,0.0737,0.0473,0.0001,0.2435,0.1094,0.0139,0.0005,0.1178,0.0197])
        emp_targets = np.array([emp_targets['emp_8001_target'],emp_targets['emp_8005_target'],emp_targets['emp_8013_target'],emp_targets['emp_8014_target'],emp_targets['emp_8019_target'],emp_targets['emp_8031_target'],emp_targets['emp_8035_target'],emp_targets['emp_8039_target'],emp_targets['emp_8047_target'],emp_targets['emp_8059_target'],emp_targets['emp_8123_target']])
        county_id = np.array([8001,8005,8013,8014,8019,8031,8035,8039,8047,8059,8123])
        targets = pd.DataFrame({'county_id':county_id,'resunit_target':resunit_targets,'hh_target':hh_targets,'emp_target':emp_targets,'nonres_target':nonres_targets})
        delta = calibration_configuration['coefficient_step_size']
        margin = calibration_configuration['match_target_within']
        iterations = calibration_configuration['iterations']
        hh_submodels = ['hh_location_1', 'hh_location_2', 'hh_location_3', 'hh_location_4', 'hh_location_5']
        emp_submodels = ['emp_location_1', 'emp_location_2', 'emp_location_3', 'emp_location_4', 'emp_location_5', 'emp_location_6']
        
        for it in range(iterations):
            dset = dataset.DRCOGDataset(os.path.join(misc.data_dir(),'drcog.h5'))

    ########SIMULATION
            logger.log_status('Model simulation...')
            
            title = name  ###use this to write out output to csv
            for scenario in [self.scenario,]: #'low_impact','scen0'
                print 'Running scenario: ' + scenario
                import time
                seconds_start = time.time()
                print seconds_start
                import numpy as np, pandas as pd, os, statsmodels.api as sm
                import synthicity.urbansim.interaction as interaction
                from synthicity.utils import misc
                import dataset, copy, math
                if fixed_seed:
                    logger.log_status('Running with fixed random seed.')
                    np.random.seed(random_seed)
                first_year = base_year
                last_year = forecast_year
                summary = {'employment':[],'households':[],'non_residential_sqft':[],'residential_units':[],'price':[]}
            #     dset = dataset.DRCOGDataset(os.path.join(misc.data_dir(),'drcog.h5'))  ####Note commented out since we do estimation in the same notebook
                coeff_store_path = os.path.join(misc.data_dir(),'coeffs.h5')
                coeff_store = pd.HDFStore(coeff_store_path)
                dset.coeffs = coeff_store.coeffs.copy()
                coeff_store.close()
                for sim_year in range(first_year,last_year+1):
                    print 'Simulating year ' + str(sim_year)

                #####Variable calculations
                    #VARIABLE LIBRARY
                    #parcel
                    p = dset.fetch('parcels')
                    p['in_denver'] = (p.county_id==8031).astype('int32')
                    p['ln_dist_rail'] = p.dist_rail.apply(np.log1p)
                    p['cherry_creek_school_district'] = (p.school_district==8).astype('int32')
                    #building
                    b = dset.fetch('buildings',building_sqft_per_job_table=elcm_configuration['building_sqft_per_job_table'],bsqft_job_scaling=elcm_configuration['scaling_factor'])
                    b = b[['building_type_id','improvement_value','land_area','non_residential_sqft','parcel_id','residential_units','sqft_per_unit','stories','tax_exempt','year_built','bldg_sq_ft','unit_price_non_residential','unit_price_residential','building_sqft_per_job','non_residential_units','base_year_jobs','all_units']]
                    b['zone_id'] = p.zone_id[b.parcel_id].values
                    b['county_id'] = p.county_id[b.parcel_id].values
                    b['townhome'] = (b.building_type_id==24).astype('int32')
                    b['multifamily'] = (np.in1d(b.building_type_id,[2,3])).astype('int32')
                    b['office'] = (b.building_type_id==5).astype('int32')
                    b['retail_or_restaurant'] = (np.in1d(b.building_type_id,[17,18])).astype('int32')
                    b['industrial_building'] = (np.in1d(b.building_type_id,[9,22])).astype('int32')
                    b['county8001'] = (b.county_id==8001).astype('int32')
                    b['county8005'] = (b.county_id==8005).astype('int32')
                    b['county8013'] = (b.county_id==8013).astype('int32')
                    b['county8014'] = (b.county_id==8014).astype('int32')
                    b['county8019'] = (b.county_id==8019).astype('int32')
                    b['county8031'] = (b.county_id==8031).astype('int32')
                    b['county8035'] = (b.county_id==8035).astype('int32')
                    b['county8039'] = (b.county_id==8039).astype('int32')
                    b['county8047'] = (b.county_id==8047).astype('int32')
                    b['county8059'] = (b.county_id==8059).astype('int32')
                    b['county8123'] = (b.county_id==8123).astype('int32')
                    #b['btype_dplcm'] = 1*(b.building_type_id==2) + 2*(b.building_type_id==3) + 3*(b.building_type_id==20) + 4*(b.building_type_id==24) + 6*np.invert(np.in1d(b.building_type_id,[2,3,20,24]))
                    b['btype_hlcm'] = 1*(b.building_type_id==2) + 2*(b.building_type_id==3) + 3*(b.building_type_id==20) + 4*np.invert(np.in1d(b.building_type_id,[2,3,20]))
                    #household
                    hh_estim = dset.fetch('households_for_estimation')
                    hh_estim['tenure'] = 1
                    hh_estim.tenure[hh_estim.own>1] = 2
                    hh_estim['income']=0
                    hh_estim.income[hh_estim.income_group==1] = 7500
                    hh_estim.income[hh_estim.income_group==2] = 17500
                    hh_estim.income[hh_estim.income_group==3] = 25000
                    hh_estim.income[hh_estim.income_group==4] = 35000
                    hh_estim.income[hh_estim.income_group==5] = 45000
                    hh_estim.income[hh_estim.income_group==6] = 55000
                    hh_estim.income[hh_estim.income_group==7] = 67500
                    hh_estim.income[hh_estim.income_group==8] = 87500
                    hh_estim.income[hh_estim.income_group==9] = 117500
                    hh_estim.income[hh_estim.income_group==10] = 142500
                    hh_estim.income[hh_estim.income_group==11] = 200000
                    hh = dset.fetch('households')
                    for table in [hh_estim, hh]:
                        choosers = table
                        choosers['zone_id'] = b.zone_id[choosers.building_id].values
                        choosers['building_type_id'] = b.building_type_id[choosers.building_id].values
                        choosers['county_id'] = b.county_id[choosers.building_id].values
                        choosers['btype'] = 1*(choosers.building_type_id==2) + 2*(choosers.building_type_id==3) + 3*(choosers.building_type_id==20) + 4*np.invert(np.in1d(choosers.building_type_id,[2,3,20]))
                        choosers['income_3_tenure'] = 1 * (choosers.income < 60000)*(choosers.tenure == 1) + 2 * np.logical_and(choosers.income >= 60000, choosers.income < 120000)*(choosers.tenure == 1) + 3*(choosers.income >= 120000)*(choosers.tenure == 1) + 4*(choosers.income < 40000)*(choosers.tenure == 2) + 5*(choosers.income >= 40000)*(choosers.tenure == 2)
                    #establishment
                    e = dset.fetch('establishments')
                    e['zone_id'] = b.zone_id[e.building_id].values
                    e['county_id'] = b.county_id[e.building_id].values
                    e['sector_id_six'] = 1*(e.sector_id==61) + 2*(e.sector_id==71) + 3*np.in1d(e.sector_id,[11,21,22,23,31,32,33,42,48,49]) + 4*np.in1d(e.sector_id,[7221,7222,7224]) + 5*np.in1d(e.sector_id,[44,45,7211,7212,7213,7223]) + 6*np.in1d(e.sector_id,[51,52,53,54,55,56,62,81,92])
                    #zone
                    z = dset.fetch('zones')
                    z['zonal_hh'] = hh.groupby('zone_id').size()
                    z['zonal_emp'] = e.groupby('zone_id').employees.sum()
                    z['zonal_pop'] = hh.groupby('zone_id').persons.sum()
                    z['residential_units_zone'] = b.groupby('zone_id').residential_units.sum()
                    z['non_residential_sqft_zone'] = b.groupby('zone_id').non_residential_sqft.sum()
                    z['percent_sf'] = b[b.btype_hlcm==3].groupby('zone_id').residential_units.sum()*100.0/(b.groupby('zone_id').residential_units.sum())
                    z['ln_avg_unit_price_zone'] = b.groupby('zone_id').unit_price_residential.mean().apply(np.log1p)
                    z['ln_avg_nonres_unit_price_zone'] = b.groupby('zone_id').unit_price_non_residential.mean().apply(np.log1p)
                    z['median_age_of_head'] = hh.groupby('zone_id').age_of_head.median()
                    z['mean_income'] = hh.groupby('zone_id').income.mean()
                    z['median_year_built'] = b.groupby('zone_id').year_built.median().astype('int32')
                    z['median_yearbuilt_post_1990'] = (b.groupby('zone_id').year_built.median()>1990).astype('int32')
                    z['median_yearbuilt_pre_1950'] = (b.groupby('zone_id').year_built.median()<1950).astype('int32')
                    z['percent_hh_with_child'] = hh[hh.children>0].groupby('zone_id').size()*100.0/z.zonal_hh
                    z['percent_renter_hh_in_zone'] = hh[hh.tenure==2].groupby('zone_id').size()*100.0/z.zonal_hh
                    z['percent_younghead'] = hh[hh.age_of_head<30].groupby('zone_id').size()*100.0/z.zonal_hh
                    z['average_resunit_size'] = b.groupby('zone_id').sqft_per_unit.mean()
                    z['zone_contains_park'] = (p[p.lu_type_id==14].groupby('zone_id').size()>0).astype('int32')
                    z['emp_sector1'] = e[e.sector_id_six==1].groupby('zone_id').employees.sum()
                    z['emp_sector2'] = e[e.sector_id_six==2].groupby('zone_id').employees.sum()
                    z['emp_sector3'] = e[e.sector_id_six==3].groupby('zone_id').employees.sum()
                    z['emp_sector4'] = e[e.sector_id_six==4].groupby('zone_id').employees.sum()
                    z['emp_sector5'] = e[e.sector_id_six==5].groupby('zone_id').employees.sum()
                    z['emp_sector6'] = e[e.sector_id_six==6].groupby('zone_id').employees.sum()
                    z['ln_jobs_within_45min'] = dset.compute_range(z.zonal_emp,45.0).apply(np.log1p)
                    z['ln_jobs_within_30min'] = dset.compute_range(z.zonal_emp,45.0).apply(np.log1p)
                    z['ln_pop_within_20min'] = dset.compute_range(z.zonal_pop,20.0).apply(np.log1p)
                    z['ln_emp_sector1_within_15min'] = dset.compute_range(z.emp_sector1,15.0).apply(np.log1p)
                    z['ln_emp_sector2_within_15min'] = dset.compute_range(z.emp_sector2,15.0).apply(np.log1p)
                    z['ln_emp_sector3_within_15min'] = dset.compute_range(z.emp_sector3,15.0).apply(np.log1p)
                    z['ln_emp_sector4_within_15min'] = dset.compute_range(z.emp_sector4,15.0).apply(np.log1p)
                    z['ln_emp_sector5_within_15min'] = dset.compute_range(z.emp_sector5,15.0).apply(np.log1p)
                    z['ln_emp_sector6_within_15min'] = dset.compute_range(z.emp_sector6,15.0).apply(np.log1p)
                    #merge parcels with zones
                    pz = pd.merge(p,z,left_on='zone_id',right_index=True)
                    #merge buildings with parcels/zones
                    bpz = pd.merge(b,pz,left_on='parcel_id',right_index=True)
                    bpz['residential_units_capacity'] = bpz.parcel_sqft/1500 - bpz.residential_units
                    bpz.residential_units_capacity[bpz.residential_units_capacity<0] = 0
                    dset.d['buildings'] = bpz
                    
                    #Record pre-demand model zone-level household/job totals
                    hh_zone1 = hh.groupby('zone_id').size()
                    emp_zone1 = e.groupby('zone_id').employees.sum()
                    
                    #Record base values for temporal comparison
                    if sim_year==first_year:
                        summary['employment'].append(e[e.building_id>0].employees.sum())
                        summary['households'].append(len(hh[hh.building_id>0].building_id))
                        summary['non_residential_sqft'].append(b.non_residential_sqft.sum())
                        summary['residential_units'].append(b.residential_units.sum())
                        print len(dset.households.index)
                        print dset.establishments.employees.sum()
                        print dset.buildings.residential_units.sum()
                        print dset.buildings.non_residential_sqft.sum()
                        base_hh_county = hh.groupby('county_id').size()
                        base_emp_county = e.groupby('county_id').employees.sum()
                        base_ru_county = b.groupby('county_id').residential_units.sum()
                        base_nr_county = b.groupby('county_id').non_residential_sqft.sum()
                    ##Estimate REPM instead of loading from CSV because it is so fast
                    if (sim_year==first_year) and core_components_to_run['Price']:
                        print 'Price estim'
                        logger.log_status('REPM estimation.')
                        buildings = dset.fetch('buildings')
                        buildings = buildings[(buildings.improvement_value>20000)*(np.in1d(buildings.building_type_id,[5,8,11,16,17,18,21,23,9,22,2,3,20,24]))]
                        output_csv, output_title, coeff_name, output_varname = ["drcog-coeff-hedonic.csv","DRCOG HEDONIC MODEL","price_%s","price"]
                        ind_vars2 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        ind_vars3 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        ind_vars5 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        ind_vars8 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        ind_vars9 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        ind_vars11 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        ind_vars16 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        ind_vars17 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        ind_vars18 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        ind_vars20 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        ind_vars21 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        ind_vars22 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        ind_vars23 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        ind_vars24 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        segments = buildings.groupby('building_type_id')
                        for name, segment in segments:
                            if name == 2:
                                indvars = ind_vars2
                            if name == 3:
                                indvars = ind_vars3
                            if name == 5:
                                indvars = ind_vars5
                            if name == 8:
                                indvars = ind_vars8
                            if name == 9:
                                indvars = ind_vars9
                            if name == 11:
                                indvars = ind_vars11
                            if name == 16:
                                indvars = ind_vars16
                            if name == 17:
                                indvars = ind_vars17
                            if name == 18:
                                indvars = ind_vars18
                            if name == 20:
                                indvars = ind_vars20
                            if name == 21:
                                indvars = ind_vars21
                            if name == 22:
                                indvars = ind_vars22
                            if name == 23:
                                indvars = ind_vars23
                            if name == 24:
                                indvars = ind_vars24
                            if name in [5,8,11,16,17,18,21,23,9,22]:
                                segment = segment[segment.unit_price_non_residential>0]
                                depvar = segment['unit_price_non_residential'].apply(np.log)
                            else:
                                segment = segment[segment.unit_price_residential>0]
                                depvar = segment['unit_price_residential'].apply(np.log)
                            est_data = pd.DataFrame(index=segment.index)
                            for varname in indvars:
                                est_data[varname] = segment[varname]
                            est_data = est_data.fillna(0)
                            est_data = sm.add_constant(est_data,prepend=False)
                            tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
                            print "Estimating hedonic for %s with %d observations" % (name,len(segment.index))
                            print est_data.describe()

                            model = sm.OLS(depvar,est_data)
                            results = model.fit()
                            print results.summary()

                            tmp_outcsv = output_csv%name
                            tmp_outtitle = output_title%name
                            misc.resultstocsv((results.rsquared,results.rsquared_adj),est_data.columns,
                                                zip(results.params,results.bse,results.tvalues),tmp_outcsv,hedonic=1,
                                                tblname=output_title)
                            dset.store_coeff(tmp_coeffname,results.params.values,results.params.index)
                            
            ############     ELCM SIMULATION
                    if core_components_to_run['ELCM']:
                        logger.log_status('ELCM simulation.')
                        depvar = 'building_id'
                        simulation_table = 'establishments'
                        output_csv, output_title, coeff_name, output_varname = ("drcog-coeff-elcm-%s.csv","DRCOG EMPLOYMENT LOCATION CHOICE MODELS (%s)","emp_location_%s","establishment_building_ids")
                        agents_groupby_txt = ['sector_id_six',]
                        #dset.establishments['home_based_status']=0 #########
                        
                        #if scenario == 'baseline':
                        if employment_transition['Enabled']:
                            logger.log_status('Running employment transition model.')
                            ct = dset.fetch(employment_transition['control_totals_table'])
                            ct["total_number_of_jobs"] = (ct["total_number_of_jobs"]*employment_transition['scaling_factor']).astype('int32')
                            new_jobs = {"table": "dset.establishments","writetotmp": "establishments","model": "transitionmodel","first_year": 2010,"control_totals": "dset.%s"%employment_transition['control_totals_table'],
                                        "geography_field": "building_id","amount_field": "total_number_of_jobs","size_field":"employees"}
                            import synthicity.urbansim.transitionmodel as transitionmodel
                            transitionmodel.simulate(dset,new_jobs,year=sim_year,show=True)
                        
                        year = sim_year
                        choosers = dset.fetch(simulation_table)
                        #     rate_table = dset.annual_job_relocation_rates
                        #     rate_table = rate_table*.1
                        #     rate_field = "job_relocation_probability"
                        #     movers = dset.relocation_rates(choosers,rate_table,rate_field)
                        #     choosers[depvar].ix[movers] = -1
                        movers = choosers[choosers[depvar]==-1]
                        print "Total new agents and movers = %d" % len(movers.index)
                        alternatives = dset.buildings[(dset.buildings.non_residential_sqft>0)]
                        #alternatives['building_sqft_per_job'] = 250
                        alternatives['job_spaces'] = alternatives.non_residential_sqft/alternatives.building_sqft_per_job
                        empty_units = alternatives.job_spaces.sub(choosers.groupby('building_id').employees.sum(),fill_value=0).astype('int')
                        alts = alternatives.ix[empty_units.index]
                        alts["supply"] = empty_units
                        lotterychoices = True
                        pdf = pd.DataFrame(index=alts.index)
                        segments = movers.groupby(agents_groupby_txt)
                        
                        ind_vars1=['ln_jobs_within_30min','ln_avg_nonres_unit_price_zone','median_year_built','residential_units_zone','ln_pop_within_20min','office','retail_or_restaurant','industrial_building','employees_x_non_residential_sqft_zone','ln_emp_sector1_within_15min']+['county8001','county8005','county8013','county8014','county8019','county8031','county8035','county8039','county8047','county8059','county8123']
                        ind_vars2=['ln_jobs_within_30min','ln_avg_nonres_unit_price_zone','median_year_built','residential_units_zone','ln_pop_within_20min','office','retail_or_restaurant','industrial_building','employees_x_non_residential_sqft_zone','ln_emp_sector2_within_15min']+['county8001','county8005','county8013','county8014','county8019','county8031','county8035','county8039','county8047','county8059','county8123']
                        ind_vars3=['ln_jobs_within_30min','ln_avg_nonres_unit_price_zone','median_year_built','residential_units_zone','ln_pop_within_20min','office','retail_or_restaurant','industrial_building','employees_x_non_residential_sqft_zone','ln_emp_sector3_within_15min']+['county8001','county8005','county8013','county8014','county8019','county8031','county8035','county8039','county8047','county8059','county8123']
                        ind_vars4=['ln_jobs_within_30min','ln_avg_nonres_unit_price_zone','median_year_built','residential_units_zone','ln_pop_within_20min','office','retail_or_restaurant','industrial_building','employees_x_non_residential_sqft_zone','ln_emp_sector4_within_15min']+['county8001','county8005','county8013','county8014','county8019','county8031','county8035','county8039','county8047','county8059','county8123']
                        ind_vars5=['ln_jobs_within_30min','ln_avg_nonres_unit_price_zone','median_year_built','residential_units_zone','ln_pop_within_20min','office','retail_or_restaurant','industrial_building','employees_x_non_residential_sqft_zone','ln_emp_sector5_within_15min']+['county8001','county8005','county8013','county8014','county8019','county8031','county8035','county8039','county8047','county8059','county8123']
                        ind_vars6=['ln_jobs_within_30min','ln_avg_nonres_unit_price_zone','median_year_built','residential_units_zone','ln_pop_within_20min','office','retail_or_restaurant','industrial_building','employees_x_non_residential_sqft_zone','ln_emp_sector6_within_15min']+['county8001','county8005','county8013','county8014','county8019','county8031','county8035','county8039','county8047','county8059','county8123']

                        for name, segment in segments:
                            if name == 1:
                                ind_vars = ind_vars1 
                            if name == 2:
                                ind_vars = ind_vars2
                            if name == 3:
                                ind_vars = ind_vars3
                            if name == 4:
                                ind_vars = ind_vars4
                            if name == 5:
                                ind_vars = ind_vars5
                            if name == 6:
                                ind_vars = ind_vars6
                             
                            segment = segment.head(1)
                            name_coeff= str(name)
                            name = str(name)
                            tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name_coeff
                            SAMPLE_SIZE = alts.index.size 
                            numchoosers = segment.shape[0]
                            numalts = alts.shape[0]
                            sample = np.tile(alts.index.values,numchoosers)
                            alts_sample = alts #sample#alternatives
                            alts_sample['join_index'] = np.repeat(segment.index,SAMPLE_SIZE)
                            alts_sample = pd.merge(alts_sample,segment,left_on='join_index',right_index=True,suffixes=('','_r'))
                            chosen = np.zeros((numchoosers,SAMPLE_SIZE))
                            chosen[:,0] = 1
                            sample, alternative_sample, est_params = sample, alts_sample, ('mnl',chosen)
                            ##Define interaction variables here
                            alternative_sample['employees_x_non_residential_sqft_zone'] = ((alternative_sample.employees)*alternative_sample.non_residential_sqft_zone).apply(np.log1p)
                            est_data = pd.DataFrame(index=alternative_sample.index)
                            for varname in ind_vars:
                                est_data[varname] = alternative_sample[varname]
                            est_data = est_data.fillna(0)
                            data = est_data
                            data = data.as_matrix()
                            coeff = dset.load_coeff(tmp_coeffname)
                            probs = interaction.mnl_simulate(data,coeff,numalts=SAMPLE_SIZE,returnprobs=1)
                            pdf['segment%s'%name] = pd.Series(probs.flatten(),index=alts.index) 
                                
                        new_homes = pd.Series(np.ones(len(movers.index))*-1,index=movers.index)
                        mask = np.zeros(len(alts.index),dtype='bool')
                        
                        for name, segment in segments:
                            name = str(name)
                            print "Assigning units to %d agents of segment %s" % (len(segment.index),name)
                            p=pdf['segment%s'%name].values
                            #p=pdf['segment%s'%name].values
                            def choose(p,mask,alternatives,segment,new_homes,minsize=None):
                                p = copy.copy(p)
                                p[alternatives.supply<minsize] = 0
                                #print "Choosing from %d nonzero alts" % np.count_nonzero(p)
                                try: 
                                  indexes = np.random.choice(len(alternatives.index),len(segment.index),replace=False,p=p/p.sum())
                                except:
                                  print "WARNING: not enough options to fit agents, will result in unplaced agents"
                                  return mask,new_homes
                                new_homes.ix[segment.index] = alternatives.index.values[indexes]
                                alternatives["supply"].ix[alternatives.index.values[indexes]] -= minsize
                                return mask,new_homes
                            tmp = segment['employees']
                            #tmp /= 100.0 ##If scaling demand amount is desired
                            for name, subsegment in reversed(list(segment.groupby(tmp.astype('int')))):
                                #print "Running subsegment with size = %s, num agents = %d" % (name, len(subsegment.index))
                                mask,new_homes = choose(p,mask,alts,subsegment,new_homes,minsize=int(name))
                        
                        build_cnts = new_homes.value_counts()  #num estabs place in each building
                        print "Assigned %d agents to %d locations with %d unplaced" % (new_homes.size,build_cnts.size,build_cnts.get(-1,0))
                        
                        table = dset.establishments # need to go back to the whole dataset
                        table[depvar].ix[new_homes.index] = new_homes.values.astype('int32')
                        dset.store_attr(output_varname,year,copy.deepcopy(table[depvar]))
                            
            #################     HLCM simulation
                    if core_components_to_run['HLCM']:
                        logger.log_status('HLCM simulation.')
                        #################     HLCM simulation
                        depvar = 'building_id'
                        simulation_table = 'households'
                        output_csv, output_title, coeff_name, output_varname = ("drcog-coeff-hlcm-%s.csv","DRCOG HOUSEHOLD LOCATION CHOICE MODELS (%s)","hh_location_%s","household_building_ids")
                        agents_groupby_txt = ['income_3_tenure',]
                        
                        #if scenario == 'baseline':
                        if household_transition['Enabled']:
                            logger.log_status('Running household transition model.')
                            ct = dset.fetch(household_transition['control_totals_table'])
                            ct["total_number_of_households"] = (ct["total_number_of_households"]*household_transition['scaling_factor']).astype('int32')
                            new_hhlds = {"table": "dset.households","writetotmp": "households","model": "transitionmodel","first_year": 2010,"control_totals": "dset.%s"%household_transition['control_totals_table'],
                                         "geography_field": "building_id","amount_field": "total_number_of_households"}
                            import synthicity.urbansim.transitionmodel as transitionmodel
                            transitionmodel.simulate(dset,new_hhlds,year=sim_year,show=True,subtract=True)
                        
                        year = sim_year
                        choosers = dset.fetch(simulation_table)
                        
                        if household_relocation['Enabled']:
                            rate_table = dset.store[household_relocation['relocation_rates_table']].copy()
                            rate_field = "probability_of_relocating"
                            rate_table[rate_field] = rate_table[rate_field]*.5*household_relocation['scaling_factor']
                            movers = dset.relocation_rates(choosers,rate_table,rate_field)
                            choosers[depvar].ix[movers] = -1
                            
                        movers = choosers[choosers[depvar]==-1]
                        print "Total new agents and movers = %d" % len(movers.index)
                        alternatives = dset.buildings[(dset.buildings.residential_units>0)]
                        empty_units = dset.buildings[(dset.buildings.residential_units>0)].residential_units.sub(choosers.groupby('building_id').size(),fill_value=0)
                        empty_units = empty_units[empty_units>0].order(ascending=False)
                        alternatives = alternatives.ix[np.repeat(empty_units.index,empty_units.values.astype('int'))]
                        alts = alternatives
                        pdf1 = pd.DataFrame(index=alts.index)
                        pdf2 = pd.DataFrame(index=alts.index) 
                        pdf3 = pd.DataFrame(index=alts.index)
                        pdf4 = pd.DataFrame(index=alts.index)
                        pdf5 = pd.DataFrame(index=alts.index)

                        segments = movers.groupby(agents_groupby_txt)

                        ind_vars1=['ln_dist_rail','ln_avg_unit_price_zone','median_age_of_head','mean_income','median_yearbuilt_post_1990','median_yearbuilt_pre_1950','proportion_hh_contain_kids_if_hh_contains_kids','percent_renter_hh_in_zone',
                                   'townhome','multifamily','cherry_creek_school_district','zone_contains_park','percent_younghead_if_younghead','ln_jobs_within_30min']+['county8001','county8005','county8013','county8014','county8019','county8031','county8035','county8039','county8047','county8059','county8123']
                        ind_vars2=['ln_dist_rail','ln_avg_unit_price_zone','median_age_of_head','mean_income','median_yearbuilt_post_1990','median_yearbuilt_pre_1950','proportion_hh_contain_kids_if_hh_contains_kids','percent_renter_hh_in_zone',
                                   'townhome','multifamily','cherry_creek_school_district','zone_contains_park','percent_younghead_if_younghead','ln_jobs_within_30min']+['county8001','county8005','county8013','county8014','county8019','county8031','county8035','county8039','county8047','county8059','county8123']
                        ind_vars3=['ln_dist_rail','ln_avg_unit_price_zone','median_age_of_head','mean_income','median_yearbuilt_post_1990','median_yearbuilt_pre_1950','proportion_hh_contain_kids_if_hh_contains_kids','percent_renter_hh_in_zone',
                                   'townhome','multifamily','cherry_creek_school_district','zone_contains_park','percent_younghead_if_younghead','ln_jobs_within_30min']+['county8001','county8005','county8013','county8014','county8019','county8031','county8035','county8039','county8047','county8059','county8123']
                        ind_vars4=['ln_dist_rail','ln_avg_unit_price_zone','median_age_of_head','mean_income','median_yearbuilt_post_1990','median_yearbuilt_pre_1950','proportion_hh_contain_kids_if_hh_contains_kids','percent_renter_hh_in_zone',
                                   'townhome','multifamily','cherry_creek_school_district','zone_contains_park','percent_younghead_if_younghead','ln_jobs_within_30min']+['county8001','county8005','county8013','county8014','county8019','county8031','county8035','county8039','county8047','county8059','county8123']
                        ind_vars5=['ln_dist_rail','ln_avg_unit_price_zone','median_age_of_head','mean_income','median_yearbuilt_post_1990','median_yearbuilt_pre_1950','proportion_hh_contain_kids_if_hh_contains_kids','percent_renter_hh_in_zone',
                                   'townhome','multifamily','cherry_creek_school_district','zone_contains_park','percent_younghead_if_younghead','ln_jobs_within_30min']+['county8001','county8005','county8013','county8014','county8019','county8031','county8035','county8039','county8047','county8059','county8123']

                        for name, segment in segments:
                            if name == 1:
                                ind_vars = ind_vars1
                            if name == 2:
                                ind_vars = ind_vars2
                            if name == 3:
                                ind_vars = ind_vars3
                            if name == 4:
                                ind_vars = ind_vars4
                            if name == 5:
                                ind_vars = ind_vars5
                            segment = segment.head(1)
                            name_coeff = str(name)
                            name = str(name)
                            tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name_coeff
                            SAMPLE_SIZE = alts.index.size 
                            numchoosers = segment.shape[0]
                            numalts = alts.shape[0]
                            sample = np.tile(alts.index.values,numchoosers)
                            alts_sample = alts
                            alts_sample['join_index'] = np.repeat(segment.index,SAMPLE_SIZE)
                            alts_sample = pd.merge(alts_sample,segment,left_on='join_index',right_index=True,suffixes=('','_r'))
                            chosen = np.zeros((numchoosers,SAMPLE_SIZE))
                            chosen[:,0] = 1
                            sample, alternative_sample, est_params = sample, alts_sample, ('mnl',chosen)
                            ##Interaction variables defined here
                            alternative_sample['proportion_hh_contain_kids_if_hh_contains_kids'] = ((alternative_sample.children>0)*alternative_sample.percent_hh_with_child)
                            alternative_sample['percent_younghead_if_younghead'] = ((alternative_sample.age_of_head<30)*alternative_sample.percent_younghead)
                            est_data = pd.DataFrame(index=alternative_sample.index)
                            for varname in ind_vars:
                                est_data[varname] = alternative_sample[varname]
                            est_data = est_data.fillna(0)
                            data = est_data
                            data = data.as_matrix()
                            coeff = dset.load_coeff(tmp_coeffname)
                            probs = interaction.mnl_simulate(data,coeff,numalts=SAMPLE_SIZE,returnprobs=1)
                            if int(name_coeff) == 1:
                                pdf1['segment%s'%name] = pd.Series(probs.flatten(),index=alts.index)  
                            if int(name_coeff) == 2:
                                pdf2['segment%s'%name] = pd.Series(probs.flatten(),index=alts.index) 
                            if int(name_coeff) == 3:
                                pdf3['segment%s'%name] = pd.Series(probs.flatten(),index=alts.index) 
                            if int(name_coeff) == 4:
                                pdf4['segment%s'%name] = pd.Series(probs.flatten(),index=alts.index) 
                            if int(name_coeff) == 5:
                                pdf5['segment%s'%name] = pd.Series(probs.flatten(),index=alts.index) 

                        new_homes = pd.Series(np.ones(len(movers.index))*-1,index=movers.index)
                        for name, segment in segments:
                            name_coeff = str(name)
                            name = str(name)
                            if int(name_coeff) == 1:
                                p=pdf1['segment%s'%name].values
                                mask = np.zeros(len(alts.index),dtype='bool')
                            if int(name_coeff) == 2:
                                p=pdf2['segment%s'%name].values 
                                mask = np.zeros(len(alts.index),dtype='bool')
                            if int(name_coeff) == 3:
                                p=pdf3['segment%s'%name].values 
                                mask = np.zeros(len(alts.index),dtype='bool')
                            if int(name_coeff) == 4:
                                p=pdf4['segment%s'%name].values
                                mask = np.zeros(len(alts.index),dtype='bool')
                            if int(name_coeff) == 5:
                                p=pdf5['segment%s'%name].values
                                mask = np.zeros(len(alts.index),dtype='bool')
                            print "Assigning units to %d agents of segment %s" % (len(segment.index),name)
                         
                            def choose(p,mask,alternatives,segment,new_homes,minsize=None):
                                p = copy.copy(p)
                                p[mask] = 0 # already chosen
                                print 'num alts'
                                print len(alternatives.index)
                                print 'num agents'
                                print len(segment.index)
                                try: 
                                  indexes = np.random.choice(len(alternatives.index),len(segment.index),replace=False,p=p/p.sum())
                                except:
                                  print "WARNING: not enough options to fit agents, will result in unplaced agents"
                                  return mask,new_homes
                                new_homes.ix[segment.index] = alternatives.index.values[indexes]
                                mask[indexes] = 1
                              
                                return mask,new_homes
                            if int(name_coeff) == 1:
                                mask,new_homes = choose(p,mask,alts,segment,new_homes)
                            if int(name_coeff) == 2:
                                mask,new_homes = choose(p,mask,alts,segment,new_homes)
                            if int(name_coeff) == 3:
                                mask,new_homes = choose(p,mask,alts,segment,new_homes)
                            if int(name_coeff) == 4:
                                mask,new_homes = choose(p,mask,alts,segment,new_homes)
                            if int(name_coeff) == 5:
                                mask,new_homes = choose(p,mask,alts,segment,new_homes)
                            
                        build_cnts = new_homes.value_counts()  #num households place in each building
                        print "Assigned %d agents to %d locations with %d unplaced" % (new_homes.size,build_cnts.size,build_cnts.get(-1,0))

                        table = dset.households # need to go back to the whole dataset
                        table[depvar].ix[new_homes.index] = new_homes.values.astype('int32')
                        dset.store_attr(output_varname,year,copy.deepcopy(table[depvar]))

                        
            ############     REPM SIMULATION   ####Uncomment this stuff when it works with the proforma developer model (There's that weird interaction where non-res proforma crashes with price models on)
                    # if core_components_to_run['Price']:
                        # logger.log_status('REPM simulation.')
                        # year = sim_year
                        # buildings = dset.fetch('buildings')
                        # buildings = buildings[np.in1d(buildings.building_type_id,[5,8,11,16,17,18,21,23,9,22,2,3,20,24])]
                        # output_csv, output_title, coeff_name = ["drcog-coeff-hedonic.csv","DRCOG HEDONIC MODEL","price_%s"]
                        # ind_vars2 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        # ind_vars3 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        # ind_vars5 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        # ind_vars8 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        # ind_vars9 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        # ind_vars11 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        # ind_vars16 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        # ind_vars17 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        # ind_vars18 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        # ind_vars20 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        # ind_vars21 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        # ind_vars22 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        # ind_vars23 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]
                        # ind_vars24 = ['parcel_sqft','dist_bus','dist_rail','land_value','residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','in_denver',]

                        # simrents = []
                        # segments = buildings.groupby('building_type_id')
                        # for name, segment in segments:
                            # if name == 2:
                                # indvars = ind_vars2
                            # if name == 3:
                                # indvars = ind_vars3
                            # if name == 5:
                                # indvars = ind_vars5
                            # if name == 8:
                                # indvars = ind_vars8
                            # if name == 9:
                                # indvars = ind_vars9
                            # if name == 11:
                                # indvars = ind_vars11
                            # if name == 16:
                                # indvars = ind_vars16
                            # if name == 17:
                                # indvars = ind_vars17
                            # if name == 18:
                                # indvars = ind_vars18
                            # if name == 20:
                                # indvars = ind_vars20
                            # if name == 21:
                                # indvars = ind_vars21
                            # if name == 22:
                                # indvars = ind_vars22
                            # if name == 23:
                                # indvars = ind_vars23
                            # if name == 24:
                                # indvars = ind_vars24
                            # est_data = pd.DataFrame(index=segment.index)
                            # for varname in indvars:
                                # est_data[varname] = segment[varname]
                            # est_data = est_data.fillna(0)
                            # est_data = sm.add_constant(est_data,prepend=False)
                            # tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
                            # print "Generating rents on %d buildings" % (est_data.shape[0])
                            # vec = dset.load_coeff(tmp_coeffname)
                            # vec = np.reshape(vec,(vec.size,1))
                            # rents = est_data.dot(vec).astype('f4')
                            # rents = rents.apply(np.exp)
                            # simrents.append(rents[rents.columns[0]])
                        # if name in [5,8,11,16,17,18,21,23,9,22]:
                            # output_varname = 'unit_price_non_residential'
                        # else:
                            # output_varname = 'unit_price_residential'
                        # simrents = pd.concat(simrents)
                        # dset.buildings[output_varname] = simrents.reindex(dset.buildings.index)
                        # dset.store_attr(output_varname,year,simrents)
                        
                
                    ############     DEVELOPER SIMULATION
                    if core_components_to_run['Developer']:
                    
                    ############     PROFORMA SIMULATION
                        if developer_configuration['Proforma']:
                            logger.log_status('Proforma simulation.')
                            #Record post-demand-model change in zone-level household/job totals
                            hh = dset.fetch('households')
                            e = dset.fetch('establishments')
                            b = dset.fetch('buildings')
                            p = dset.fetch('parcels')
                            b['zone_id'] = p.zone_id[b.parcel_id].values
                            e['zone_id'] = b.zone_id[e.building_id].values
                            hh['zone_id'] = b.zone_id[hh.building_id].values
                            hh_zone2 = hh.groupby('zone_id').size()
                            emp_zone2 = e.groupby('zone_id').employees.sum()
                            hh_zone_diff = hh_zone2 - hh_zone1
                            emp_zone_diff = emp_zone2 - emp_zone1
                            
                            #####Get the user inputted zone args
                            if developer_configuration['zonal_levers']:
                                zone_args = pd.read_csv(os.path.join(misc.data_dir(),'devmodal_zone_args.csv')).set_index('zone_id')
                            else:
                                zone_args = None
                            
                            # import pandas as pd, numpy as np
                            # import time, os
                            # from synthicity.utils import misc
                            from urbandeveloper import spotproforma, developer
                            # import dataset
                            # dset = dataset.DRCOGDataset(os.path.join(misc.data_dir(),'drcog.h5'))
                            dev = spotproforma.Developer(profit_factor=developer_configuration['profit_factor'])
                            year = sim_year
                            def get_possible_rents_by_use(dset, zone_args):
                                parcels = dset.parcels 
                                buildings = dset.buildings
                                buildings = buildings[['building_type_id','improvement_value','land_area','non_residential_sqft','parcel_id','residential_units','sqft_per_unit','stories','tax_exempt','year_built','bldg_sq_ft','unit_price_non_residential','unit_price_residential','building_sqft_per_job','non_residential_units','base_year_jobs','all_units']]
                                buildings['zone_id'] = parcels.zone_id[buildings.parcel_id].values
                                res_buildings = buildings[buildings.unit_price_residential>0]
                                
                                nonres_buildings = buildings[buildings.unit_price_non_residential>0]
                                nonres_buildings_office = nonres_buildings[nonres_buildings.building_type_id==5]
                                nonres_buildings_retail = nonres_buildings[np.in1d(nonres_buildings.building_type_id,[17,18])]
                                nonres_buildings_industrial = nonres_buildings[np.in1d(nonres_buildings.building_type_id,[9,22])]
                                res_buildings['resprice_sqft'] = res_buildings.unit_price_residential/res_buildings.sqft_per_unit
                                zonal_resprice_sqft = res_buildings.groupby('zone_id').resprice_sqft.mean()
                                zonal_nonresprice_office = nonres_buildings_office.groupby('zone_id').unit_price_non_residential.mean()
                                zonal_nonresprice_retail = nonres_buildings_retail.groupby('zone_id').unit_price_non_residential.mean()
                                zonal_nonresprice_industrial = nonres_buildings_industrial.groupby('zone_id').unit_price_non_residential.mean()
                                zonal_resrent = zonal_resprice_sqft/17.9  
                                zonal_nonresrent_office = zonal_nonresprice_office/17.9
                                zonal_nonresrent_retail = zonal_nonresprice_retail/17.9
                                zonal_nonresrent_industrial = zonal_nonresprice_industrial/17.9
                                
                                if zone_args is not None:
                                    zonal_resrent = zonal_resrent * zone_args.res_price_factor
                                    zonal_nonresrent_office = zonal_nonresprice_office * zone_args.nonres_price_factor
                                    zonal_nonresrent_retail = zonal_nonresprice_retail * zone_args.nonres_price_factor
                                    zonal_nonresrent_industrial = zonal_nonresprice_industrial * zone_args.nonres_price_factor
                                    zonal_avg_rents = pd.DataFrame({'resrent':zonal_resrent,'nonresrent_office':zonal_nonresrent_office,'nonresrent_retail':zonal_nonresrent_retail,'nonresrent_industrial':zonal_nonresrent_industrial,'cost_factor':zone_args.cost_factor,'allowable_density_factor':zone_args.allowable_density_factor})
                                else:
                                    zonal_avg_rents = pd.DataFrame({'resrent':zonal_resrent,'nonresrent_office':zonal_nonresrent_office,'nonresrent_retail':zonal_nonresrent_retail,'nonresrent_industrial':zonal_nonresrent_industrial})
                                
                                avgrents = pd.merge(parcels,zonal_avg_rents,left_on='zone_id',right_index=True,how='left')
                                avgrents['residential'] = avgrents.resrent
                                avgrents['office'] = avgrents.nonresrent_office
                                avgrents['retail'] = avgrents.nonresrent_retail
                                avgrents['industrial'] = avgrents.nonresrent_industrial
                                if zone_args is not None:
                                    avgrents = avgrents[['residential','office','retail','industrial','cost_factor','allowable_density_factor']]
                                else:
                                    avgrents = avgrents[['residential','office','retail','industrial']]
                                avgrents = avgrents.fillna(1)  ###1 is a low value, and also happens to be a neutral cost-scaling factor placeholder
                                return avgrents
                                
                            parcels = dset.fetch('parcels')
                            buildings = dset.fetch('buildings')
                            avgrents = get_possible_rents_by_use(dset, zone_args)

                            buildings['bldg_sq_ft'] = buildings.non_residential_sqft + buildings.residential_units*buildings.sqft_per_unit
                            buildings['impval'] = buildings.non_residential_sqft*buildings.unit_price_non_residential + buildings.residential_units*buildings.unit_price_residential
                            far_predictions = pd.DataFrame(index=parcels.index)
                            far_predictions['total_sqft'] = buildings.groupby('parcel_id').bldg_sq_ft.sum()
                            far_predictions['total_sqft'] = far_predictions.total_sqft.fillna(0)
                            far_predictions['current_yearly_rent_buildings'] = buildings.groupby('parcel_id').impval.sum()/17.9
                            far_predictions['current_yearly_rent_buildings'] = far_predictions.current_yearly_rent_buildings.fillna(0)
                            far_predictions.current_yearly_rent_buildings = far_predictions.current_yearly_rent_buildings * developer_configuration['land_property_acquisition_cost_factor']  
                            if zone_args is not None:
                                far_predictions.current_yearly_rent_buildings = avgrents.cost_factor*far_predictions.current_yearly_rent_buildings ##Cost scaling happens here
                            far_predictions['parcelsize'] = parcels.parcel_sqft
                            #far_predictions.parcelsize[far_predictions.parcelsize<300] = 300 # some parcels have unrealisticly small sizes ##Keep this or just filter out?

                            # do the lookup in the developer model - this is where the profitability is computed
                            for form in spotproforma.forms.keys():
                                far_predictions[form+'_feasiblefar'], far_predictions[form+'_profit'] = \
                                        dev.lookup(form,avgrents[spotproforma.uses].as_matrix(),far_predictions.current_yearly_rent_buildings,far_predictions.parcelsize)
                            # we now have a far prediction per parcel

                            zoning = dset.fetch('zoning')
                            fars = dset.fetch('fars')
                            
                            max_parcel_sqft = 200000
                            max_far_field = developer_configuration['max_allowable_far_field_name']
                            if max_far_field not in parcels.columns:
                                parcels = pd.merge(parcels,fars,left_on='far_id',right_index=True)
                                if developer_configuration['enforce_environmental_constraints']:
                                    parcels[max_far_field] = parcels[max_far_field]*(1 - parcels.prop_constrained) #Adjust allowable FAR to account for undevelopable proportion of parcel land
                                if developer_configuration['enforce_ugb']:
                                    parcels[max_far_field][parcels.in_ugb==0] = parcels[max_far_field][parcels.in_ugb==0] * developer_configuration['outside_ugb_allowable_density']
                                if developer_configuration['uga_policies']:
                                    parcels[max_far_field][parcels.in_uga==1] = parcels[max_far_field][parcels.in_ugb==1] * developer_configuration['inside_uga_allowable_density']
                                parcels[max_far_field][parcels.parcel_sqft<developer_configuration['min_lot_sqft']] = 0
                                parcels[max_far_field][parcels.parcel_sqft>max_parcel_sqft] = 0
                            if 'type1' not in parcels.columns:
                                parcels = pd.merge(parcels,zoning,left_on='zoning_id',right_index=True)
                                
                            ##Scale allowable FARs here if needed
                            if zone_args is not None:
                                parcels[max_far_field] = parcels[max_far_field]*avgrents.allowable_density_factor

                            type_d = { 
                            'residential': [2,3,20,24],
                            'industrial': [9,22],
                            'retail': [17,18],
                            'office': [5],
                            #'mixedresidential': [11],  ###Turning off for now
                            #'mixedoffice': [999], ##does not exist in drcog
                            #Howbout hotels?  We need hotels (btype 22/23
                            }

                            # we have zoning by like 16 building types and rents/far predictions by 4 building types
                            # so we have to convert one into the other - would probably be better to have rents
                            # segmented by the same 16 building types if we had good observations for that
                            parcel_predictions = pd.DataFrame(index=parcels.index)
                            for typ, btypes in type_d.iteritems():
                                for btype in btypes:

                                    # three questions - 1) is type allowed 2) what FAR is allowed 3) is it supported by rents
                                    if developer_configuration['enforce_allowable_use_constraints']:
                                        tmp = parcels[parcels['type%d'%btype]==1][[max_far_field]] # is type allowed
                                        far_predictions['type%d_zonedfar'%btype] = tmp[max_far_field] # at what far
                                    else:
                                        far_predictions['type%d_zonedfar'%btype] = parcels[max_far_field]
                                    
                                    # merge zoning with feasibility
                                    tmp = pd.merge(tmp,far_predictions[[typ+'_feasiblefar']],left_index=True,right_index=True,how='left').set_index(tmp.index)
                                    
                                    # min of zoning and feasibility
                                    parcel_predictions[btype] = pd.Series(np.minimum(tmp[max_far_field],tmp[typ+'_feasiblefar']),index=tmp.index) 
                                    
                            parcel_predictions = parcel_predictions.dropna(how='all').sort_index(axis=1)
                            print "Average rents\n", avgrents.describe()
                            print "Feasibility\n", far_predictions.describe()
                            print "Restricted to zoning\n", parcel_predictions.describe()
                            print "Feasible square footage (in millions)"
                            for col in parcel_predictions.columns: 
                                print col, (parcel_predictions[col]*far_predictions.parcelsize).sum()/1000000.0
                            ####Sampling parcels for developers to consider.  Comment next 3 lines if no sample.
                            p_sample_proportion = .5
                            parcel_predictions = parcel_predictions.ix[np.random.choice(parcel_predictions.index, int(len(parcel_predictions.index)*p_sample_proportion),replace=False)]
                            parcel_predictions.index.name = 'parcel_id'
                            parcel_predictions.to_csv(os.path.join(misc.data_dir(),'parcel_predictions.csv'),index_col='parcel_id',float_format="%.2f")
                            far_predictions.to_csv(os.path.join(misc.data_dir(),'far_predictions.csv'),index_col='parcel_id',float_format="%.2f")
                            print "Finished developer", time.ctime()
                            print hh_zone_diff.sum()
                            print emp_zone_diff.sum()
                            
                            #####CALL TO THE DEVELOPER
                            newbuildings = developer.run(dset,hh_zone_diff,emp_zone_diff,year=year,min_building_sqft=developer_configuration['min_building_sqft'],min_lot_sqft=developer_configuration['min_lot_sqft'],max_lot_sqft=max_parcel_sqft,zone_args=zone_args)

                            ##When net residential units is less than 0, do we need to implement building demolition?
                            newbuildings = newbuildings[['building_type_id','building_sqft','residential_units','lot_size']]
                            newbuildings = newbuildings.reset_index()
                            newbuildings.columns = ['parcel_id','building_type_id','bldg_sq_ft','residential_units','land_area']
                            newbuildings.residential_units = newbuildings.residential_units.astype('int32')
                            newbuildings.land_area = newbuildings.land_area.astype('int32')
                            newbuildings.building_type_id = newbuildings.building_type_id.astype('int32')
                            newbuildings.parcel_id = newbuildings.parcel_id.astype('int32')
                            newbuildings.bldg_sq_ft = np.round(newbuildings.bldg_sq_ft).astype('int32')

                            newbuildings['non_residential_sqft'] = 0
                            newbuildings.non_residential_sqft[newbuildings.residential_units == 0] = newbuildings.bldg_sq_ft
                            newbuildings['improvement_value'] = (newbuildings.non_residential_sqft*100 + newbuildings.residential_units*100000).astype('int32')
                            newbuildings['sqft_per_unit'] = 0
                            newbuildings.sqft_per_unit[newbuildings.residential_units>0] = 1000
                            newbuildings['stories'] = np.ceil(newbuildings.bldg_sq_ft*1.0/newbuildings.land_area).astype('int32')
                            newbuildings['tax_exempt'] = 0
                            newbuildings['year_built'] = year
                            newbuildings['unit_price_residential'] = 0.0
                            newbuildings.unit_price_residential[newbuildings.residential_units>0] = buildings[buildings.unit_price_residential>0].unit_price_residential.median()
                            newbuildings['unit_price_non_residential'] = 0.0
                            newbuildings.unit_price_non_residential[newbuildings.non_residential_sqft>0] = buildings[buildings.unit_price_non_residential>0].unit_price_non_residential.median()
                            newbuildings.unit_price_residential[newbuildings.residential_units>0]  = 100000.0
                            newbuildings.unit_price_non_residential[newbuildings.residential_units==0] = 100.0
                            newbuildings['building_sqft_per_job'] = 250.0  #####Need to replace with observed
                            newbuildings['non_residential_units'] = (newbuildings.non_residential_sqft/newbuildings.building_sqft_per_job).fillna(0)
                            newbuildings['base_year_jobs'] = 0.0
                            newbuildings['all_units'] = newbuildings.non_residential_units + newbuildings.residential_units 

                            newbuildings.non_residential_sqft = newbuildings.non_residential_sqft.astype('int32')
                            newbuildings.tax_exempt = newbuildings.tax_exempt.astype('int32')
                            newbuildings.year_built = newbuildings.year_built.astype('int32')
                            newbuildings.sqft_per_unit = newbuildings.sqft_per_unit.astype('int32')
                            newbuildings = newbuildings.set_index(np.arange(len(newbuildings.index))+np.amax(buildings.index.values)+1)

                            buildings = buildings[['building_type_id','improvement_value','land_area','non_residential_sqft','parcel_id','residential_units','sqft_per_unit','stories','tax_exempt','year_built','bldg_sq_ft','unit_price_non_residential','unit_price_residential','building_sqft_per_job','non_residential_units','base_year_jobs','all_units']]
                            dset.d['buildings'] = pd.concat([buildings,newbuildings])
                            
                    ############     RDPLCM SIMULATION
                        else:
                            logger.log_status('RDPLCM simulation.')
                            depvar = 'parcel_id'
                            simulation_table = 'buildings'
                            output_csv, output_title, coeff_name, output_varname = ("drcog-coeff-rdplcm-%s.csv","DRCOG RES DEVPROJECT LOCATION CHOICE MODELS (%s)","rdp_location_%s","resbuilding_parcel_ids")
                            agents_groupby_txt = ['building_type_id',]
                            
                            #if scenario == 'baseline':
                            logger.log_status('Running development project transition model.')
                            ct = dset.fetch(developer_configuration['vacancy_rates_table'])
                            ct[sim_year] = ct[sim_year]*developer_configuration['vacancy_scaling_factor']
                            new_dev = {"table": "dset.buildings","writetotmp": "buildings","model": "transitionmodel","first_year": 2010,"vacancy_targets": {"targets": "dset.%s"%developer_configuration['vacancy_rates_table'],
                                       "supply": "dset.buildings.groupby('building_type_id').all_units.sum()",
                                       "demands": ["dset.households.groupby(dset.buildings.building_type_id[dset.households.building_id].values).size()",
                                       "dset.establishments.groupby(dset.buildings.building_type_id[dset.establishments.building_id].values).employees.sum()"]},"size_field": "all_units","geography_field": "parcel_id"}
                            import synthicity.urbansim.transitionmodel as transitionmodel
                            transitionmodel.simulate(dset,new_dev,year=sim_year,show=True)
                            
                            year = sim_year
                            choosers = dset.fetch(simulation_table)
                            choosers = choosers[(np.in1d(choosers.building_type_id,[2,3,20,24]))]
                            movers = choosers[choosers[depvar]==-1]
                            print "Total new agents and movers = %d" % len(movers.index)
                            
                            #ALTS
                            alternatives = dset.parcels
                            alts1 = alternatives
                            alts2 = alternatives
                            alts3 = alternatives
                            alts4 = alternatives
                            
                            alternatives = dset.parcels[(dset.parcels.parcel_sqft>2000)]
                            # alts1 = alternatives
                            # alts2 = alternatives
                            # alts3 = alternatives
                            # alts4 = alternatives
                            alternatives['ru_spaces'] = alternatives.parcel_sqft/2000
                            empty_units = alternatives.ru_spaces.sub(choosers.groupby('parcel_id').residential_units.sum(),fill_value=0).astype('int')
                            alts = alternatives.ix[empty_units.index]
                            alts["supply"] = empty_units
                            lotterychoices = True
                            pdf = pd.DataFrame(index=alts.index)
                            segments = movers.groupby(agents_groupby_txt)
                            
                            ind_vars1=['parcel_sqft','dist_bus','dist_rail','land_value']
                            ind_vars2=['parcel_sqft','dist_bus','dist_rail','land_value']
                            ind_vars3=['parcel_sqft','dist_bus','dist_rail','land_value']
                            ind_vars4=['parcel_sqft','dist_bus','dist_rail','land_value']
                            
                            for name, segment in segments:
                                if name == 1:
                                    ind_vars = ind_vars1 
                                if name == 2:
                                    ind_vars = ind_vars2
                                if name == 3:
                                    ind_vars = ind_vars3
                                if name == 4:
                                    ind_vars = ind_vars4
                                 
                                segment = segment.head(1)
                                name_coeff= str(name)
                                name = str(name)
                                tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name_coeff
                                SAMPLE_SIZE = alts.index.size 
                                numchoosers = segment.shape[0]
                                numalts = alts.shape[0]
                                sample = np.tile(alts.index.values,numchoosers)
                                alts_sample = alts
                                alts_sample['join_index'] = np.repeat(segment.index,SAMPLE_SIZE)
                                alts_sample = pd.merge(alts_sample,segment,left_on='join_index',right_index=True,suffixes=('','_r'))
                                chosen = np.zeros((numchoosers,SAMPLE_SIZE))
                                chosen[:,0] = 1
                                sample, alternative_sample, est_params = sample, alts_sample, ('mnl',chosen)
                                ##Define interaction variables here
                                est_data = pd.DataFrame(index=alternative_sample.index)
                                for varname in ind_vars:
                                    est_data[varname] = alternative_sample[varname]
                                est_data = est_data.fillna(0)
                                data = est_data
                                data = data.as_matrix()
                                coeff = dset.load_coeff(tmp_coeffname)
                                probs = interaction.mnl_simulate(data,coeff,numalts=SAMPLE_SIZE,returnprobs=1)
                                pdf['segment%s'%name] = pd.Series(probs.flatten(),index=alts.index) 
                                    
                            new_homes = pd.Series(np.ones(len(movers.index))*-1,index=movers.index)
                            mask = np.zeros(len(alts.index),dtype='bool')
                            
                            for name, segment in segments:
                                name = str(name)
                                print "Assigning units to %d agents of segment %s" % (len(segment.index),name)
                                p=pdf['segment%s'%name].values
                                #p=pdf['segment%s'%name].values
                                def choose(p,mask,alternatives,segment,new_homes,minsize=None):
                                    p = copy.copy(p)
                                    p[alternatives.supply<minsize] = 0
                                    #print "Choosing from %d nonzero alts" % np.count_nonzero(p)
                                    try: 
                                      indexes = np.random.choice(len(alternatives.index),len(segment.index),replace=False,p=p/p.sum())
                                    except:
                                      print "WARNING: not enough options to fit agents, will result in unplaced agents"
                                      return mask,new_homes
                                    new_homes.ix[segment.index] = alternatives.index.values[indexes]
                                    alternatives["supply"].ix[alternatives.index.values[indexes]] -= minsize
                                    return mask,new_homes
                                tmp = segment['residential_units']
                                #tmp /= 100.0 ##If scaling demand amount is desired
                                for name, subsegment in reversed(list(segment.groupby(tmp.astype('int')))):
                                    #print "Running subsegment with size = %s, num agents = %d" % (name, len(subsegment.index))
                                    mask,new_homes = choose(p,mask,alts,subsegment,new_homes,minsize=int(name))
                            
                            build_cnts = new_homes.value_counts()  #num estabs place in each building
                            print "Assigned %d agents to %d locations with %d unplaced" % (new_homes.size,build_cnts.size,build_cnts.get(-1,0))
                            
                            table = dset.fetch(simulation_table)  # need to go back to the whole dataset, for rdplcm this is buildings
                            table[depvar].ix[new_homes.index] = new_homes.values.astype('int32')
                            dset.store_attr(output_varname,year,copy.deepcopy(table[depvar]))
                        
                            ############     NRDPLCM SIMULATION
                            logger.log_status('NRDPLCM simulation.')
                            depvar = 'parcel_id'
                            simulation_table = 'buildings'
                            output_csv, output_title, coeff_name, output_varname = ("drcog-coeff-nrdplcm-%s.csv","DRCOG NONRES DEVPROJECT LOCATION CHOICE MODELS (%s)","nrdp_location_%s","nonresbuilding_parcel_ids")
                            agents_groupby_txt = ['building_type_id',]
                            
                            year = sim_year
                            choosers = dset.fetch(simulation_table)
                            choosers = choosers[(np.in1d(choosers.building_type_id,[5,8,11,16,17,18,21,23,9,22]))] ##9 and 22 are industrial and should have different allowable parcels
                            movers = choosers[choosers[depvar]==-1]
                            print "Total new agents and movers = %d" % len(movers.index)
                            
                            #ALTS
                            alternatives = dset.parcels[(dset.parcels.parcel_sqft>5000)]
                            alternatives = alternatives.ix[np.random.choice(alternatives.index, 30000,replace=False)]
                            alternatives['nr_spaces'] = alternatives.parcel_sqft/10
                            empty_units = alternatives.nr_spaces.sub(choosers.groupby('parcel_id').non_residential_sqft.sum(),fill_value=0).astype('int')
                            alts = alternatives.ix[empty_units.index]
                            alts["supply"] = empty_units
                            lotterychoices = True
                            pdf = pd.DataFrame(index=alts.index)
                            segments = movers.groupby(agents_groupby_txt)
                            
                            ##Need to break alternatives down by allowable use during simulation
                            alts1 = alts
                            alts2 = alts
                            alts3 = alts
                            alts4 = alts
                            alts5 = alts
                            alts6 = alts
                            alts7 = alts
                            alts8 = alts
                            alts9 = alts
                            alts10 = alts
                            ind_vars1=['parcel_sqft','dist_bus','dist_rail','land_value']
                            ind_vars2=['parcel_sqft','dist_bus','dist_rail','land_value']
                            ind_vars3=['parcel_sqft','dist_bus','dist_rail','land_value']
                            ind_vars4=['parcel_sqft','dist_bus','dist_rail','land_value']
                            ind_vars5=['parcel_sqft','dist_bus','dist_rail','land_value']
                            ind_vars6=['parcel_sqft','dist_bus','dist_rail','land_value']
                            ind_vars7=['parcel_sqft','dist_bus','dist_rail','land_value']
                            ind_vars8=['parcel_sqft','dist_bus','dist_rail','land_value']
                            ind_vars9=['parcel_sqft','dist_bus','dist_rail','land_value']
                            ind_vars10=['parcel_sqft','dist_bus','dist_rail','land_value']
                            
                            for name, segment in segments:
                                if name == 5:
                                    alts = alts1
                                    ind_vars = ind_vars1
                                if name == 8:
                                    alts = alts2
                                    ind_vars = ind_vars2
                                if name == 11:
                                    alts = alts3
                                    ind_vars = ind_vars3
                                if name == 16:
                                    alts = alts4
                                    ind_vars = ind_vars4
                                if name == 17:
                                    alts = alts5
                                    ind_vars = ind_vars5
                                if name == 18:
                                    alts = alts6
                                    ind_vars = ind_vars6
                                if name == 21:
                                    alts = alts7
                                    ind_vars = ind_vars7
                                if name == 23:
                                    alts = alts8
                                    ind_vars = ind_vars8
                                if name == 9:
                                    alts = alts9
                                    ind_vars = ind_vars9
                                if name == 22:
                                    alts = alts10
                                    ind_vars = ind_vars10
                                 
                                segment = segment.head(1)
                                name_coeff= str(name)
                                name = str(name)
                                tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name_coeff
                                SAMPLE_SIZE = alts.index.size 
                                numchoosers = segment.shape[0]
                                numalts = alts.shape[0]
                                sample = np.tile(alts.index.values,numchoosers)
                                alts_sample = alts
                                alts_sample['join_index'] = np.repeat(segment.index,SAMPLE_SIZE)
                                alts_sample = pd.merge(alts_sample,segment,left_on='join_index',right_index=True,suffixes=('','_r'))
                                chosen = np.zeros((numchoosers,SAMPLE_SIZE))
                                chosen[:,0] = 1
                                sample, alternative_sample, est_params = sample, alts_sample, ('mnl',chosen)
                                ##Define interaction variables here
                                est_data = pd.DataFrame(index=alternative_sample.index)
                                for varname in ind_vars:
                                    est_data[varname] = alternative_sample[varname]
                                est_data = est_data.fillna(0)
                                data = est_data
                                data = data.as_matrix()
                                coeff = dset.load_coeff(tmp_coeffname)
                                probs = interaction.mnl_simulate(data,coeff,numalts=SAMPLE_SIZE,returnprobs=1)
                                pdf['segment%s'%name] = pd.Series(probs.flatten(),index=alts.index) 
                                    
                            new_homes = pd.Series(np.ones(len(movers.index))*-1,index=movers.index)
                            mask = np.zeros(len(alts.index),dtype='bool')
                            
                            for name, segment in segments:
                                name = str(name)
                                print "Assigning units to %d agents of segment %s" % (len(segment.index),name)
                                p=pdf['segment%s'%name].values
                                #p=pdf['segment%s'%name].values
                                def choose(p,mask,alternatives,segment,new_homes,minsize=None):
                                    p = copy.copy(p)
                                    p[alternatives.supply<minsize] = 0
                                    #print "Choosing from %d nonzero alts" % np.count_nonzero(p)
                                    try: 
                                      indexes = np.random.choice(len(alternatives.index),len(segment.index),replace=False,p=p/p.sum())
                                    except:
                                      print "WARNING: not enough options to fit agents, will result in unplaced agents"
                                      return mask,new_homes
                                    new_homes.ix[segment.index] = alternatives.index.values[indexes]
                                    alternatives["supply"].ix[alternatives.index.values[indexes]] -= minsize
                                    return mask,new_homes
                                tmp = segment['non_residential_sqft']
                                #tmp /= 100.0 ##If scaling demand amount is desired
                                for name, subsegment in reversed(list(segment.groupby(tmp.astype('int')))):
                                    #print "Running subsegment with size = %s, num agents = %d" % (name, len(subsegment.index))
                                    mask,new_homes = choose(p,mask,alts,subsegment,new_homes,minsize=int(name))
                            
                            build_cnts = new_homes.value_counts()  #num estabs place in each building
                            print "Assigned %d agents to %d locations with %d unplaced" % (new_homes.size,build_cnts.size,build_cnts.get(-1,0))
                            
                            table = dset.fetch(simulation_table)  # need to go back to the whole dataset, for rdplcm this is buildings
                            table[depvar].ix[new_homes.index] = new_homes.values.astype('int32')
                            dset.store_attr(output_varname,year,copy.deepcopy(table[depvar]))
                    

                    #####SUMMARIES
                    #########################
                    if export_indicators:
                        print 'Annual totals'
                        print len(dset.households.index)
                        print dset.establishments.employees.sum()
                        print dset.buildings.residential_units.sum()
                        print dset.buildings.non_residential_sqft.sum()
                        
                        if sim_year == last_year:
                            print summary
                            b = dset.fetch('buildings')
                            e = dset.fetch('establishments')
                            hh = dset.fetch('households')
                            p = dset.fetch('parcels')
                            b['county_id'] = p.county_id[b.parcel_id].values
                            hh['county_id'] = b.county_id[hh.building_id].values
                            e['county_id'] = b.county_id[e.building_id].values
                            sim_hh_county = hh.groupby('county_id').size()
                            sim_emp_county = e.groupby('county_id').employees.sum()
                            sim_ru_county = b.groupby('county_id').residential_units.sum()
                            sim_nr_county = b.groupby('county_id').non_residential_sqft.sum()
                            hh_diff_county = sim_hh_county - base_hh_county
                            emp_diff_county = sim_emp_county - base_emp_county
                            ru_diff_county = sim_ru_county - base_ru_county
                            nr_diff_county = sim_nr_county - base_nr_county
                            print 'Household diffs'
                            print hh_diff_county
                            print 'Employment diffs'
                            print emp_diff_county
                            print 'Resunit diffs'
                            print ru_diff_county
                            print 'NR sqft diffs'
                            print nr_diff_county
                            logger.log_status('Household diffs')
                            logger.log_status(hh_diff_county)
                            logger.log_status(hh_diff_county.sum())
                            logger.log_status('Employment diffs')
                            logger.log_status(emp_diff_county)
                            logger.log_status(emp_diff_county.sum())
                            logger.log_status('Resunit diffs')
                            logger.log_status(ru_diff_county)
                            logger.log_status(ru_diff_county.sum())
                            logger.log_status('NR sqft diffs')
                            logger.log_status(nr_diff_county)
                            logger.log_status(nr_diff_county.sum())
                            ###Calibration
                            prop_growth_emp = emp_diff_county*1.0/emp_diff_county.sum()
                            prop_growth_hh = hh_diff_county*1.0/hh_diff_county.sum()
                            prop_growth_ru = ru_diff_county*1.0/ru_diff_county.sum()
                            prop_growth_nr = nr_diff_county*1.0/nr_diff_county.sum()
                            i = 0;j = 0;k = 0;m = 0
                            for x in targets.county_id.values:
                                cid = int(x)
                                print cid
                                prop_ru = prop_growth_ru[cid]
                                prop_hh = prop_growth_hh[cid]
                                prop_emp = prop_growth_emp[cid]
                                prop_nonres = prop_growth_nr[cid]
                                print 'ru prop is ' + str(prop_ru)
                                print 'nsqft prop is ' + str(prop_nonres)
                                print 'hh prop is ' + str(prop_hh)
                                print 'emp prop is ' + str(prop_emp)
                                target_ru = targets.resunit_target[targets.county_id==cid].values[0]
                                target_hh = targets.hh_target[targets.county_id==cid].values[0]
                                target_emp = targets.emp_target[targets.county_id==cid].values[0]
                                target_nonres = targets.nonres_target[targets.county_id==cid].values[0]
                                print 'ru target is ' + str(target_ru)
                                print 'nsqft target is ' + str(target_nonres)
                                print 'hh target is ' + str(target_hh)
                                print 'emp target is ' + str(target_emp)
                                varname = 'county%s' % (cid)
                                print varname
                                # if (prop_ru > (target_ru - margin)) and (prop_ru < (target_ru + margin)):
                                    # print 'NO ru action.'
                                    # i = i + 1
                                # elif math.isnan(prop_ru) or (prop_ru < target_ru):
                                    # for submodel in ru_submodels:
                                        # dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] = dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] + delta
                                    # print 'ru action is PLUS'
                                # elif prop_ru > target_ru:
                                    # for submodel in ru_submodels:
                                        # dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] = dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] - delta
                                    # print 'ru action is MINUS'
                                    
                                if (prop_hh > (target_hh - margin)) and (prop_hh < (target_hh + margin)):
                                    print 'NO hh action.'
                                    j = j + 1
                                elif math.isnan(prop_hh) or (prop_hh < target_hh):
                                    for submodel in hh_submodels:
                                        dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] = dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] + delta
                                    print 'hh action is PLUS'
                                elif prop_hh > target_hh:
                                    for submodel in hh_submodels:
                                        dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] = dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] - delta
                                    print 'hh action is MINUS'
                                    
                                if (prop_emp > (target_emp - margin)) and (prop_emp < (target_emp + margin)):
                                    print 'NO emp action.'
                                    k = k + 1
                                elif math.isnan(prop_emp) or (prop_emp < target_emp):
                                    for submodel in emp_submodels:
                                        dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] = dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] + delta
                                    print 'emp action is PLUS'
                                elif prop_emp > target_emp:
                                    for submodel in emp_submodels:
                                        dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] = dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] - delta
                                    print 'emp action is MINUS'
                                    
                                # if (prop_nonres > (target_nonres - margin)) and (prop_nonres < (target_nonres + margin)):
                                    # print 'NO nonres action.'
                                    # m = m + 1
                                # elif math.isnan(prop_nonres) or (prop_nonres < target_nonres):
                                    # for submodel in nr_submodels:
                                        # dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] = dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] + delta
                                    # print 'nonres action is PLUS'
                                # elif prop_nonres > target_nonres:
                                    # for submodel in nr_submodels:
                                        # dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] = dset.coeffs[(submodel, 'coeffs')][dset.coeffs[(submodel,'fnames')]==varname] - delta
                                    # print 'nonres action is MINUS'
                            print i,j,k,m
                            logger.log_status('Number of hh county targets met: %s' % j)
                            logger.log_status('Number of emp county targets met: %s' % k)
                            ###Save calibrated coefficients at the end of each iteration
                            coeff_store_path = os.path.join(misc.data_dir(),'coeffs.h5')
                            coeff_store = pd.HDFStore(coeff_store_path)
                            coeff_store['coeffs'] = dset.coeffs
                            coeff_store.close()
                elapsed = time.time() - seconds_start
                print "TOTAL elapsed time: " + str(elapsed) + " seconds."
                
            print 'Done running all three scenarios.'