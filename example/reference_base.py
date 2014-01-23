title = '_ref_base'
for scenario in ['baseline',]: #'low_impact','scen0'
    print 'Running scenario: ' + scenario
    import time
    seconds_start = time.time()
    print seconds_start
    import numpy as np, pandas as pd, os, statsmodels.api as sm
    import synthicity.urbansim.interaction as interaction
    from synthicity.utils import misc
    import dataset, copy, math
    np.random.seed(1)
    first_year = 2000
    last_year = 2035
    summary = {'employment':[],'households':[],'non_residential_sqft':[],'residential_units':[],'price':[]}
    dset = dataset.PARISDataset(os.path.join(misc.data_dir(),'paris.h5'))
    
    for sim_year in range(first_year,last_year+1):
        print 'Simulating year ' + str(sim_year)
        
    #####Variable calculations
        b = dset.fetch('buildings')
        if sim_year==first_year:
            b = pd.merge(b,dset.store.building_sqft_per_job,left_index=True,right_index=True)
        b=b[['building_type_id','non_residential_sqft','non_residential_sqft_capacity','price','residential_units','residential_units_capacity','zone_id','building_sqft_per_job','estim_index']]
    
        b['ln_average_res_price1'] = b.price[b.building_type_id==1].apply(np.log1p)
        b['ln_average_res_price2'] = b.price[b.building_type_id==2].apply(np.log1p)
        b['ln_average_res_price3'] = b.price[b.building_type_id==3].apply(np.log1p)
        b['ln_average_res_price4'] = b.price[b.building_type_id==4].apply(np.log1p)
        b['ln_average_nonres_price6'] = b.price[b.building_type_id==6].apply(np.log1p)
        b['price1'] = b.price[b.building_type_id==1]
        b['price2'] = b.price[b.building_type_id==2]
        b['price3'] = b.price[b.building_type_id==3]
        b['price4'] = b.price[b.building_type_id==4]
        b['ln_residential_units_owner_house'] = b.residential_units[b.building_type_id==1].apply(np.log1p)
        b['ln_residential_units_owner_flat'] = b.residential_units[b.building_type_id==2].apply(np.log1p)
        b['ln_residential_units_renter_house'] = b.residential_units[b.building_type_id==3].apply(np.log1p)
        b['ln_residential_units_renter_flat'] = b.residential_units[b.building_type_id==4].apply(np.log1p)
        #household
        hh = dset.fetch('households')
        hh['zone_id'] = b.zone_id[hh.building_id].values
        if sim_year==first_year:
            hh['foreign'] = (hh.race_id==1).astype('int32')
            hh['income_cat'] = 1*(hh.lincomepc<=9.9) + 2*(hh.lincomepc>9.9)*(hh.lincomepc<=10.3) + 3*(hh.lincomepc>10.3)
            hh['high_inc'] = (hh.income_cat==3).astype('int32')
            hh['mid_inc'] = (hh.income_cat==2).astype('int32')
            hh['low_inc'] = (hh.income_cat==1).astype('int32')
            hh['age_cat'] = 1*(hh.age_of_head<=35) + 2*(hh.age_of_head>35)*(hh.age_of_head<=60) + 3*(hh.age_of_head>60)
            hh['btype_tenure'] = 1*np.in1d(hh.hh_type,[1,5,9,13,17,21]) + 2*np.in1d(hh.hh_type,[2,6,10,14,18,22]) + 3*np.in1d(hh.hh_type,[3,7,11,15,19,23]) + 4*np.in1d(hh.hh_type,[4,8,12,16,20,24])
            hh['hhsize3plus'] = (hh.persons>2).astype('int32')
            hh['hhsize2'] = (hh.persons==2).astype('int32')
            hh['young'] = (hh.age_cat==1).astype('int32')
            hh['middle_age'] = (hh.age_cat==2).astype('int32')
            hh['old'] = (hh.age_cat==3).astype('int32')
            hh['with_child'] = (hh.children>0).astype('int32')
            hh['with_car'] = (hh.cars>0).astype('int32')
            hh['previous_county'] = 1+1*(hh.previous_dpt==75)+2*(np.in1d(hh.previous_dpt,[92,93,94]))
        #establishment
        e = dset.fetch('establishments')
        e['zone_id'] = b.zone_id[e.building_id].values
        if sim_year==first_year:
            e['more_than_10_employees'] = (e.employees>10).astype('int32')
            e['less_than_10_employees'] = (e.employees<10).astype('int32')
        e1998 = dset.fetch('establishments1998')
        if sim_year==first_year:
            e1998['zone_id'] = b.zone_id[e1998.building_id].values
        
        #zone
        z = dset.fetch('zones')
        z['mean_household_size'] = hh.groupby('zone_id').persons.mean()
        z['mean_age_of_head'] = hh.groupby('zone_id').age_of_head.mean()
        zonal_household_totals = hh.groupby('zone_id').building_id.count()
        z['total_households'] = hh.groupby('zone_id').building_id.count()
        z['total_persons'] = hh.groupby('zone_id').persons.sum()
        z['total_residential_units'] = b.groupby('zone_id').residential_units.sum()
        z['total_nonresidential_sqft'] = b.groupby('zone_id').non_residential_sqft.sum()
        z['percent_foreigners'] = hh.groupby('zone_id').foreign.sum()*100.0/(zonal_household_totals)
        z['percent_low_income'] = hh[hh.income_cat==1].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['percent_mid_income'] = hh[hh.income_cat==2].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['percent_high_income'] = hh[hh.income_cat==3].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['percent_hhsize3plus'] = hh[hh.hhsize3plus==1].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['percent_hhsize2'] = hh[hh.hhsize2==1].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['percent_young'] = hh[hh.age_cat==1].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['percent_middle_age'] = hh[hh.age_cat==2].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['percent_old'] = hh[hh.age_cat==3].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['percent_with_child'] = hh[hh.children>0].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['percent_hh_zero_worker'] = hh[hh.workers==0].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['percent_hh_one_worker'] = hh[hh.workers==1].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['percent_hh_two_worker'] = hh[hh.workers==2].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['percent_hh_twoplus_workers'] = hh[hh.workers>1].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['percent_hh_threeplus_workers'] = hh[hh.workers>2].groupby('zone_id').building_id.count()*100.0/(zonal_household_totals)
        z['population_density'] = z.crrdenspop
        if scenario == 'baseline':
            travel_data = dset.store.travel_data_baseline[dset.store.travel_data_baseline.year==sim_year]
        if scenario == 'low_impact':
            travel_data = dset.store.travel_data_low_impact[dset.store.travel_data_low_impact.year==sim_year]
        if scenario == 'scen0':
            travel_data = dset.store.travel_data_scen0[dset.store.travel_data_scen0.year==sim_year]
        z['tco'] = travel_data.tco
        z['vpo'] = travel_data.vpo
        z['tcd'] = travel_data.tcd
        z['vpd'] = travel_data.vpd
        if sim_year==first_year:
            z['in_paris'] = (z.dept==75).astype('int32')
            z['in_biotech'] = (z.zgp_id==22).astype('int32')
            z['in_clichy_montfermeil'] = (z.zgp_id==21).astype('int32')
            z['in_confluence'] = (z.zgp_id==23).astype('int32')
            z['in_descartes'] = (z.zgp_id==24).astype('int32')
            z['in_le_bourget'] = (z.zgp_id==26).astype('int32')
            z['in_paris_pole'] = ((z.zgp_id>=27)*(z.zgp_id<=46)).astype('int32')
            z['in_pleyel'] = (z.zgp_id==47).astype('int32')
            z['in_roissy'] = (z.zgp_id==48).astype('int32')
            z['in_saclay'] = (z.zgp_id==49).astype('int32')
            z['in_val_de_france_gonesse'] = (z.zgp_id==50).astype('int32')
            z['in_la_defense'] = np.in1d(z.insee,[92062,92026,92050]).astype('int32') 
            z['in_new_town'] = (z.cvilnouvel>0).astype('int32')  
            z['in_paris_suburbs'] = np.in1d(z.dept_id,[92,93,94]).astype('int32') 
            z['distance_to_arterial'] = z.cdistart/1000
            z['distance_to_highway'] = z.cdisthwy/1000
            z['ln_land_area'] = z.careakm2.apply(np.log1p)
            z['cloactpotst'] = z.cnoactpotst.apply(np.log1p)
            z['tax_on_professionals'] = z.taxpro
            z['percent_education_level1'] = z.ctpniv1
            z['percent_education_level2'] = z.ctpniv2
            z['percent_education_level3'] = z.ctpniv3
            z['percent_education_level4'] = z.ctpniv4
            z['zgpgroup_id'] = 99*(z.zgp_id<21) + 21*(z.zgp_id==21) + 22*(z.zgp_id==22)+23*(z.zgp_id==23)+24*(z.zgp_id==24)+25*(z.zgp_id==25)+26*(z.zgp_id==26) +75*(z.zgp_id>26)*(z.zgp_id<47) + 47*(z.zgp_id==47) + 48*(z.zgp_id==48) + 49*(z.zgp_id==49) + 50*(z.zgp_id==50)
            z['zgpgroup21'] = (z.zgpgroup_id ==21).astype('int32')
            z['zgpgroup22'] = (z.zgpgroup_id ==22).astype('int32')
            z['zgpgroup23'] = (z.zgpgroup_id ==23).astype('int32')
            z['zgpgroup24'] = (z.zgpgroup_id ==24).astype('int32')
            z['zgpgroup25'] = (z.zgpgroup_id ==25).astype('int32')
            z['zgpgroup26'] = (z.zgpgroup_id ==26).astype('int32')
            z['zgpgroup47'] = (z.zgpgroup_id ==47).astype('int32')
            z['zgpgroup48'] = (z.zgpgroup_id ==48).astype('int32')
            z['zgpgroup49'] = (z.zgpgroup_id ==49).astype('int32')
            z['zgpgroup50'] = (z.zgpgroup_id ==50).astype('int32')
            z['zgpgroup75'] = (z.zgpgroup_id ==75).astype('int32')
            z['zgpgroup99'] = (z.zgpgroup_id ==99).astype('int32')
        z['total_employment'] = e.groupby('zone_id').employees.sum()
        z['total_employment_prev_yr'] = e1998.groupby('zone_id').employees.sum()
        z['employment_density'] = z.total_employment/z.careakm2
        z['prev_yr_empdensity_sector2'] = e1998[e1998.sector_id==2].groupby('zone_id').employees.sum()*1000/z.cnoactpotst
        z['prev_yr_empdensity_sector3'] = e1998[e1998.sector_id==3].groupby('zone_id').employees.sum()*1000/z.cnoactpotst
        z['prev_yr_empdensity_sector4'] = e1998[e1998.sector_id==4].groupby('zone_id').employees.sum()*1000/z.cnoactpotst
        z['prev_yr_empdensity_sector5'] = e1998[e1998.sector_id==5].groupby('zone_id').employees.sum()*1000/z.cnoactpotst
        z['prev_yr_empdensity_sector6'] = e1998[e1998.sector_id==6].groupby('zone_id').employees.sum()*1000/z.cnoactpotst
        z['prev_yr_empdensity_sector7'] = e1998[e1998.sector_id==7].groupby('zone_id').employees.sum()*1000/z.cnoactpotst
        z['prev_yr_empdensity_sector8'] = e1998[e1998.sector_id==8].groupby('zone_id').employees.sum()*1000/z.cnoactpotst
        z['prev_yr_empdensity_sector9'] = e1998[e1998.sector_id==9].groupby('zone_id').employees.sum()*1000/z.cnoactpotst
        z['prev_yr_empdensity_sector10'] = e1998[e1998.sector_id==10].groupby('zone_id').employees.sum()*1000/z.cnoactpotst
        z['prev_yr_empdensity_sector11'] = e1998[e1998.sector_id==11].groupby('zone_id').employees.sum()*1000/z.cnoactpotst
        
        #Reset for lag variables as simulation progresses, but before establishments get updated this year
        dset.d['establishments1998'] = e.copy()
    
        #merge buildings with zones
        dset.d['buildings'] = pd.merge(b,z,left_on='zone_id',right_index=True)
        
        #Record 1999 values for temporal comparison
        if sim_year==first_year:
            summary['employment'].append(e[e.building_id>0].employees.sum())
            summary['households'].append(len(hh[hh.building_id>0].building_id))
            summary['non_residential_sqft'].append(b.non_residential_sqft.sum())
            summary['residential_units'].append(b.residential_units.sum())
            summary['price'].append(b.price.mean())
        if sim_year==2006:
            base_emp = z.groupby('zgpgroup_id').total_employment.sum()
            base_hh = z.groupby('zgpgroup_id').total_households.sum()
            base_ru = z.groupby('zgpgroup_id').total_residential_units.sum()
            base_nr = z.groupby('zgpgroup_id').total_nonresidential_sqft.sum()
            base_pop = z.groupby('zgpgroup_id').total_persons.sum()
            base_emp_zone = z.total_employment.copy()
            base_pop_zone = z.total_persons.copy()
        
        ##Estimate REPM instead of loading from CSV because it is so fast
        if sim_year==first_year:
            buildings = dset.fetch('buildings')
            buildings = buildings[buildings.estim_index==1]
            output_csv, output_title, coeff_name, output_varname = ["paris-coeff-hedonic-%s.csv","PARIS HEDONIC MODEL (%s)","price_%s","price"]
            ind_vars1 = ['in_paris_suburbs','percent_low_income','cpraft90','cprbef15','percent_old','bati',
                         'population_density','tco',]
            ind_vars2 = ['in_paris','in_paris_suburbs','csubway9','percent_low_income','cpraft90','cprbef15','percent_old','bati',
                         'population_density','tco',]
            ind_vars3 = ['in_paris_suburbs','csubway9','percent_low_income','cpraft90','cprbef15','percent_old',
                         'population_density','tco',]
            ind_vars4 = ['in_paris','in_paris_suburbs','csubway9','percent_low_income','cpraft90','cprbef15','percent_old','bati',
                         'population_density','tco',]
            ind_vars6 = ['in_paris','in_la_defense','in_new_town','in_paris_suburbs','csubway9','percent_low_income','cpraft90','cprbef15','bati',
                         'employment_density','tax_on_professionals','tco',]
            segments = buildings.groupby('building_type_id')
            for name, segment in segments:
                if name == 1:
                    indvars = ind_vars1
                if name == 2:
                    indvars = ind_vars2
                if name == 3:
                    indvars = ind_vars3
                if name == 4:
                    indvars = ind_vars4
                if name == 6:
                    indvars = ind_vars6
                est_data = pd.DataFrame(index=segment.index)
                for varname in indvars:
                    est_data[varname] = segment[varname]
                est_data = est_data.fillna(0)
                est_data = sm.add_constant(est_data,prepend=False)
                tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
                #print tmp_coeffname
                depvar = segment['price'].apply(np.log)
                print "Estimating hedonic for %s with %d observations" % (name,len(segment.index))
                #print est_data.describe()
                model = sm.OLS(depvar,est_data)
                results = model.fit()
                #print results.summary()
                tmp_outcsv = output_csv%name
                tmp_outtitle = output_title%name
                misc.resultstocsv((results.rsquared,results.rsquared_adj),est_data.columns,
                                    zip(results.params,results.bse,results.tvalues),tmp_outcsv,hedonic=1,
                                    tblname=output_title)
                dset.store_coeff(tmp_coeffname,results.params.values,results.params.index)
                
            ##Load location choice model coefficients from csv or hdf5
            output_dir = os.path.join(os.environ['DATA_HOME'],'output')
            output_dir = os.path.join(output_dir, 'for_runs')
            coeff_store_path = os.path.join(output_dir,'coeffs.h5')
            coeff_store = pd.HDFStore(coeff_store_path)
            dset.coeffs = coeff_store.coeffs.copy()
            coeff_store.close()
            hh_submodels = ['hh_location_1', 'hh_location_2', 'hh_location_3', 'hh_location_4']
            for name in hh_submodels:
                if name == 'hh_location_3':
                    colname1 = (name,'coeffs')
                    colname2 = (name,'fnames')
                    fnames =  dset.coeffs[colname2].append(pd.Series(['not_paris_subway_stations_x_1car', 'not_paris_subway_stations_x_2pluscar'],index=[48,49]))
                    coeffs =  dset.coeffs[colname1].append(pd.Series([0.0997, 0.0997],index=[48,49]))
                    dset.store_coeff(name,coeffs.values,fnames.values)
                if name == 'hh_location_1':
                    colname1 = (name,'coeffs')
                    colname2 = (name,'fnames')
                    fnames =  dset.coeffs[colname2].append(pd.Series(['not_paris_subway_stations_x_1car', 'not_paris_subway_stations_x_2pluscar'],index=[48,49]))
                    coeffs =  dset.coeffs[colname1].append(pd.Series([0.0593, 0.0593],index=[48,49]))
                    dset.store_coeff(name,coeffs.values,fnames.values)
                if name == 'hh_location_2':
                    dset.coeffs[(name,'fnames')][46] = 'not_paris_subway_stations_x_1car'
                    dset.coeffs[(name,'fnames')][47] = 'not_paris_subway_stations_x_2pluscar'
                    dset.coeffs[(name,'coeffs')][46] = 0.0380
                    dset.coeffs[(name,'coeffs')][47] = 0.0380
                if name == 'hh_location_4':
                    dset.coeffs[(name,'fnames')][47] = 'not_paris_subway_stations_x_1car'
                    dset.coeffs[(name,'coeffs')][47] = 0.0277
                    dset.coeffs[(name,'fnames')][48] = 'not_paris_subway_stations_x_2pluscar'
                    dset.coeffs[(name,'coeffs')][48] = 0.0277
                    
    ############     SCHEDULED DEVELOPMENT EVENTS
        if scenario == 'baseline':
            sched_events = dset.fetch('scheduled_development_events_baseline')
        if scenario == 'low_impact':
            sched_events = dset.fetch('scheduled_development_events_low_impact')
        if scenario == 'scen0':
            sched_events = dset.fetch('scheduled_development_events_scen0')
        scheduled_nonres_sqft = sched_events[(sched_events.amount>0)*(sched_events.year==sim_year)*(sched_events.attribute=='non_residential_sqft')]
        print 'Added %s scheduled non-residential projects.' % len(scheduled_nonres_sqft.index)
        if len(scheduled_nonres_sqft.index)>0:
            scheduled_nonres_sqft = scheduled_nonres_sqft.reset_index().groupby('building_id').amount.sum()
            dset.buildings.non_residential_sqft[np.in1d(dset.buildings.index,scheduled_nonres_sqft.index)] = dset.buildings.non_residential_sqft[np.in1d(dset.buildings.index,scheduled_nonres_sqft.index)] + scheduled_nonres_sqft
        scheduled_resunits = sched_events[(sched_events.amount>0)*(sched_events.year==sim_year)*(sched_events.attribute=='residential_units')]
        print 'Added %s scheduled residential projects.' % len(scheduled_resunits.index)
        if len(scheduled_resunits.index)>0:
            scheduled_resunits = scheduled_resunits.reset_index().groupby('building_id').amount.sum()
            dset.buildings.residential_units[np.in1d(dset.buildings.index,scheduled_resunits.index)] = dset.buildings.residential_units[np.in1d(dset.buildings.index,scheduled_resunits.index)] + scheduled_resunits


    ############     ELCM
        if sim_year > 2000:
            output_csv, output_title, coeff_name, output_varname = ("paris-coeff-elcm-%s.csv","PARIS EMPLOYMENT LOCATION CHOICE MODELS (%s)","emp_location_%s","establishment_building_ids")
            dset.establishments['home_based_status']=0
            if scenario == 'baseline':
                new_jobs = {"table": "dset.establishments","writetotmp": "establishments","model": "transitionmodel","first_year": 1999,"control_totals": "dset.annual_employment_control_totals_baseline",
                            "geography_field": "building_id","amount_field": "number_of_jobs","size_field":"employees"}
            if scenario == 'low_impact':
                new_jobs = {"table": "dset.establishments","writetotmp": "establishments","model": "transitionmodel","first_year": 1999,"control_totals": "dset.annual_employment_control_totals_low_impact",
                            "geography_field": "building_id","amount_field": "number_of_jobs","size_field":"employees"}
            if scenario == 'scen0':
                new_jobs = {"table": "dset.establishments","writetotmp": "establishments","model": "transitionmodel","first_year": 1999,"control_totals": "dset.annual_employment_control_totals_scen0",
                            "geography_field": "building_id","amount_field": "number_of_jobs","size_field":"employees"}
            import synthicity.urbansim.transitionmodel as transitionmodel
            transitionmodel.simulate(dset,new_jobs,year=sim_year,show=True)
            year = sim_year
            choosers = dset.fetch('establishments')
            depvar = 'building_id'
        #     rate_table = dset.annual_job_relocation_rates
        #     rate_table = rate_table*.1
        #     rate_field = "job_relocation_probability"
        #     movers = dset.relocation_rates(choosers,rate_table,rate_field)
        #     choosers[depvar].ix[movers] = -1
            movers = choosers[choosers[depvar]==-1]
            print "Total new agents and movers = %d" % len(movers.index)
            alternatives = dset.buildings[(dset.buildings.non_residential_sqft>0)*(dset.buildings.building_type_id==6)]
            alternatives['job_spaces'] = alternatives.non_residential_sqft/alternatives.building_sqft_per_job
            empty_units = alternatives.job_spaces.sub(choosers.groupby('building_id').employees.sum(),fill_value=0).astype('int')
            alts = alternatives.ix[empty_units.index]
            alts["supply"] = empty_units
            lotterychoices = True
            pdf = pd.DataFrame(index=alts.index)
            segments = movers.groupby(['sector_id','more_than_10_employees'])
            
            ind_vars1=['in_paris','distance_to_highway','total_employment_prev_yr','ctrain9','prev_yr_empdensity_sector2','prev_yr_empdensity_sector3','prev_yr_empdensity_sector4','prev_yr_empdensity_sector5',
                       'prev_yr_empdensity_sector8','population_density','tcd','vpd'] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
            ind_vars2=['in_biotech','in_la_defense','in_new_town','total_employment_prev_yr','in_clichy_montfermeil','cloactpotst','ln_nonres_price_x_more_than_10_employees','in_confluence','prev_yr_empdensity_sector2','prev_yr_empdensity_sector3',
                       'prev_yr_empdensity_sector4','prev_yr_empdensity_sector5','prev_yr_empdensity_sector8','population_density','tcd','percent_education_level1','percent_education_level2',
                       'in_descartes','in_le_bourget','in_paris_pole','in_pleyel','in_roissy','in_saclay','in_val_de_france_gonesse',
                       ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
            ind_vars3=['in_biotech','in_la_defense','in_new_town','total_employment_prev_yr','in_clichy_montfermeil','cloactpotst','ln_average_nonres_price6','in_confluence','csubway9','ctrain9',
                       'prev_yr_empdensity_sector2','prev_yr_empdensity_sector3','prev_yr_empdensity_sector4','population_density','tcd','vpd','percent_education_level1','percent_education_level2',
                       'in_descartes','in_le_bourget','in_paris_pole','in_pleyel','in_roissy','in_saclay','in_val_de_france_gonesse'
                       ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
            ind_vars4=['in_biotech','in_la_defense','in_new_town','total_employment_prev_yr','in_clichy_montfermeil','cloactpotst','ln_nonres_price_x_more_than_10_employees','in_confluence','csubway9','percent_high_income','percent_low_income',
                       'prev_yr_empdensity_sector2','prev_yr_empdensity_sector3','prev_yr_empdensity_sector4','prev_yr_empdensity_sector6','prev_yr_empdensity_sector9','population_density','tcd','vpd',
                       'in_descartes','in_le_bourget','in_paris_pole','in_pleyel','in_roissy','in_saclay','in_val_de_france_gonesse',
                       ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
            ind_vars5=['in_biotech','in_la_defense','total_employment_prev_yr','in_clichy_montfermeil','cloactpotst','ln_nonres_price_x_more_than_10_employees','in_confluence','csubway9',
                       'prev_yr_empdensity_sector2','prev_yr_empdensity_sector3','prev_yr_empdensity_sector5','population_density',
                       'in_descartes','in_le_bourget','in_paris_pole','in_pleyel','in_roissy','in_saclay','in_val_de_france_gonesse',
                       ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
            ind_vars6=['in_biotech','in_la_defense','in_new_town','total_employment_prev_yr','in_clichy_montfermeil','cloactpotst','ln_nonres_price_x_less_than_10_employees','in_confluence','csubway9','ctrain9','percent_high_income',
                       'prev_yr_empdensity_sector4','prev_yr_empdensity_sector6','prev_yr_empdensity_sector7','population_density','vpd','percent_education_level3','percent_education_level4',
                       'in_descartes','in_le_bourget','in_paris_pole','in_pleyel','in_roissy','in_saclay','in_val_de_france_gonesse',
                       ]  + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
            ind_vars7=['in_biotech','in_la_defense','in_new_town','total_employment_prev_yr','in_clichy_montfermeil','cloactpotst','in_confluence','csubway9','ctrain9','percent_high_income',
                       'prev_yr_empdensity_sector3','prev_yr_empdensity_sector6','prev_yr_empdensity_sector7','prev_yr_empdensity_sector8','prev_yr_empdensity_sector9','population_density','tcd','vpd','percent_education_level3','percent_education_level4',
                       'in_descartes','in_le_bourget','in_paris_pole','in_pleyel','in_roissy','in_saclay','in_val_de_france_gonesse',
                       ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
            ind_vars8=['in_biotech','in_la_defense','in_new_town','total_employment_prev_yr','in_clichy_montfermeil','cloactpotst','ln_nonres_price_x_less_than_10_employees','in_confluence','csubway9','ctrain9','percent_high_income','percent_low_income','percent_hh_zero_worker',
                       'prev_yr_empdensity_sector2','prev_yr_empdensity_sector7','prev_yr_empdensity_sector8','tax_on_professionals','population_density','vpd','percent_education_level3','percent_education_level4',
                       'in_descartes','in_le_bourget','in_paris_pole','in_pleyel','in_roissy','in_saclay','in_val_de_france_gonesse',
                       ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
            ind_vars9=['in_biotech','in_la_defense','in_new_town','total_employment_prev_yr','in_clichy_montfermeil','cloactpotst','ln_nonres_price_x_less_than_10_employees','in_confluence','csubway9','percent_old','percent_young','percent_high_income','percent_low_income','czfu',
                       'percent_hh_zero_worker','percent_with_child','prev_yr_empdensity_sector9','population_density','vpd','tcd',
                       'in_descartes','in_le_bourget','in_paris_pole','in_pleyel','in_roissy','in_saclay','in_val_de_france_gonesse'
                       ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
            ind_vars10=['in_biotech','in_la_defense','in_new_town','total_employment_prev_yr','in_clichy_montfermeil','cloactpotst','ln_average_nonres_price6','in_confluence','ctrain9','percent_old','percent_young','percent_high_income','percent_low_income',
                       'percent_with_child','prev_yr_empdensity_sector5','prev_yr_empdensity_sector6','prev_yr_empdensity_sector10','prev_yr_empdensity_sector11','population_density','vpd',
                       'in_descartes','in_le_bourget','in_paris_pole','in_pleyel','in_roissy','in_saclay','in_val_de_france_gonesse'
                       ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
            ind_vars11=['in_biotech','in_la_defense','in_new_town','total_employment_prev_yr','in_clichy_montfermeil','cloactpotst','ln_nonres_price_x_more_than_10_employees','in_confluence',
                       'prev_yr_empdensity_sector9','prev_yr_empdensity_sector10','prev_yr_empdensity_sector11','vpd',
                       'in_descartes','in_le_bourget','in_paris_pole','in_pleyel','in_roissy','in_saclay','in_val_de_france_gonesse'
                       ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
            
            for name, segment in segments:
                if name[0] == 1:
                    ind_vars = ind_vars1 
                if name[0] == 2:
                    ind_vars = ind_vars2
                if name[0] == 3:
                    ind_vars = ind_vars3
                if name[0] == 4:
                    ind_vars = ind_vars4
                if name[0] == 5:
                    ind_vars = ind_vars5
                if name[0] == 6:
                    ind_vars = ind_vars6
                if name[0] == 7:
                    ind_vars = ind_vars7
                if name[0] == 8:
                    ind_vars = ind_vars8
                if name[0] == 9:
                    ind_vars = ind_vars9
                if name[0] == 10:
                    ind_vars = ind_vars10
                if name[0] == 11:
                    ind_vars = ind_vars11
                 
                segment = segment.head(1)
                name_coeff= str(name[0])
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
                alternative_sample['ln_nonres_price_x_more_than_10_employees'] = (alternative_sample.ln_average_nonres_price6*alternative_sample.more_than_10_employees)
                alternative_sample['ln_nonres_price_x_less_than_10_employees'] = (alternative_sample.ln_average_nonres_price6*alternative_sample.less_than_10_employees)
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
        
        
    #################     HLCM
        output_csv, output_title, coeff_name, output_varname = ("paris-coeff-hlcm-%s.csv","PARIS HOUSEHOLD LOCATION CHOICE MODELS (%s)","hh_location_%s","household_building_ids")
        if scenario == 'baseline':
            new_hhlds = {"table": "dset.households","writetotmp": "households","model": "transitionmodel","first_year": 1999,"control_totals": "dset.annual_household_control_totals_baseline",
                         "geography_field": "building_id","amount_field": "total_number_of_households"}
        if scenario == 'low_impact':
            new_hhlds = {"table": "dset.households","writetotmp": "households","model": "transitionmodel","first_year": 1999,"control_totals": "dset.annual_household_control_totals_low_impact",
                         "geography_field": "building_id","amount_field": "total_number_of_households"}
        if scenario == 'scen0':
            new_hhlds = {"table": "dset.households","writetotmp": "households","model": "transitionmodel","first_year": 1999,"control_totals": "dset.annual_household_control_totals_scen0",
                         "geography_field": "building_id","amount_field": "total_number_of_households"}
        import synthicity.urbansim.transitionmodel as transitionmodel
        transitionmodel.simulate(dset,new_hhlds,year=sim_year,show=True,subtract=True)
        year = sim_year
        choosers = dset.fetch('households')
        depvar = 'building_id'
        rate_table = dset.annual_household_relocation_rates
        rate_table = rate_table*.1
        rate_field = "probability_of_relocating"
        movers = dset.relocation_rates(choosers,rate_table,rate_field)
        choosers[depvar].ix[movers] = -1
        movers = choosers[choosers[depvar]==-1]
        print "Total new agents and movers = %d" % len(movers.index)
        alternatives = dset.buildings[(dset.buildings.residential_units>0)]
        empty_units = dset.buildings[(dset.buildings.residential_units>0)].residential_units.sub(choosers.groupby('building_id').size(),fill_value=0)
        empty_units = empty_units[empty_units>0].order(ascending=False)
        alternatives = alternatives.ix[np.repeat(empty_units.index,empty_units.values.astype('int'))]
        alts1 = alternatives[alternatives.building_type_id==1]
        alts2 = alternatives[alternatives.building_type_id==2]
        alts3 = alternatives[alternatives.building_type_id==3]
        alts4 = alternatives[alternatives.building_type_id==4]
        pdf1 = pd.DataFrame(index=alts1.index)
        pdf2 = pd.DataFrame(index=alts2.index) 
        pdf3 = pd.DataFrame(index=alts3.index)
        pdf4 = pd.DataFrame(index=alts4.index)
        
        #segments = movers.groupby(['btype_tenure',])
        #choosers.groupby(['btype_tenure','income_cat','age_cat','sex_of_head'])
        segments = movers.groupby(['btype_tenure','low_inc','old','race_id','with_child','with_car']) 
        
        ind_vars1=["ln_residential_units_owner_house",'in_paris','in_paris_suburbs','cd_chatelet','rail_stations_x_0car','rail_stations_x_1car','rail_stations_x_2pluscar',
                   'subway_stations_x_0car','subway_stations_x_1car','subway_stations_x_2pluscar','cnoise','cpbois','cpparc_jardin','cpeau','perc_gardens_x_children',
                   'ln_price1_x_low_income','ln_price1_x_mid_income','ln_price1_x_high_income','percent_middle_age_x_high_inc','perc_foreign_x_french','perc_foreign_x_foreign','low_inc_x_percent_low_inc','mid_inc_x_percent_mid_inc',
                   'high_inc_x_percent_high_inc','hhsize2_x_percent_hhsize2','hhsize3plus_x_percent_hhsize3plus','perc_young_x_young','perc_middle_age_x_middle_age','perc_old_x_old',
                   'perc_with_child_x_child_in_hh','tco_x_0car','tco_x_1car','tco_x_2pluscar','vpo_x_0car','vpo_x_1car','vpo_x_2pluscar',
                   ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99',
                   'not_paris_subway_stations_x_2pluscar','not_paris_subway_stations_x_1car']
        ind_vars2=["ln_residential_units_owner_flat",'in_paris','in_paris_suburbs','cd_chatelet','rail_stations_x_0car','rail_stations_x_1car','rail_stations_x_2pluscar',
                   'subway_stations_x_0car','subway_stations_x_1car','subway_stations_x_2pluscar','cnoise','cpbois','cpparc_jardin','cpeau','perc_gardens_x_children',
                   'ln_price2_x_low_income','ln_price2_x_mid_income','perc_foreign_x_french','perc_foreign_x_foreign','low_inc_x_percent_low_inc','mid_inc_x_percent_mid_inc',
                   'high_inc_x_percent_high_inc','hhsize2_x_percent_hhsize2','hhsize3plus_x_percent_hhsize3plus','perc_young_x_young','perc_middle_age_x_middle_age','perc_old_x_old',
                   'perc_with_child_x_child_in_hh','tco_x_0car','tco_x_1car','tco_x_2pluscar','vpo_x_0car','vpo_x_1car','vpo_x_2pluscar',
                   ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99',
                   'not_paris_subway_stations_x_2pluscar','not_paris_subway_stations_x_1car']
        ind_vars3=["ln_residential_units_renter_house",'in_paris','in_paris_suburbs','cd_chatelet','rail_stations_x_0car','rail_stations_x_1car','rail_stations_x_2pluscar',
                   'subway_stations_x_0car','subway_stations_x_1car','subway_stations_x_2pluscar','cnoise','cpbois','cpparc_jardin','cpeau','perc_gardens_x_children',
                   'ln_price3_x_low_income','ln_price3_x_mid_income','ln_price3_x_high_income','perc_foreign_x_french','perc_foreign_x_foreign','low_inc_x_percent_low_inc','mid_inc_x_percent_mid_inc',
                   'high_inc_x_percent_high_inc','hhsize2_x_percent_hhsize2','hhsize3plus_x_percent_hhsize3plus','perc_young_x_young','perc_middle_age_x_middle_age','perc_old_x_old',
                   'perc_with_child_x_child_in_hh','percent_old_x_low_inc','tco_x_0car','tco_x_1car','tco_x_2pluscar','vpo_x_0car','vpo_x_1car','vpo_x_2pluscar',
                   ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99',
                   'not_paris_subway_stations_x_2pluscar','not_paris_subway_stations_x_1car']
        ind_vars4=["ln_residential_units_renter_flat",'in_paris','in_paris_suburbs','cd_chatelet','rail_stations_x_0car','rail_stations_x_1car','rail_stations_x_2pluscar',
                   'subway_stations_x_0car','subway_stations_x_1car','subway_stations_x_2pluscar','cnoise','cpbois','cpparc_jardin','cpeau','perc_gardens_x_children',
                   'ln_price4_x_low_income','ln_price4_x_mid_income','perc_foreign_x_french','perc_foreign_x_foreign','low_inc_x_percent_low_inc','mid_inc_x_percent_mid_inc',
                   'high_inc_x_percent_high_inc','hhsize2_x_percent_hhsize2','hhsize3plus_x_percent_hhsize3plus','perc_young_x_young','perc_middle_age_x_middle_age','perc_old_x_old',
                   'perc_with_child_x_child_in_hh','percent_middle_age_x_mid_inc','tco_x_0car','tco_x_1car','tco_x_2pluscar','vpo_x_0car','vpo_x_1car','vpo_x_2pluscar',
                   ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99',
                   'not_paris_subway_stations_x_2pluscar','not_paris_subway_stations_x_1car'] 
    
        for name, segment in segments:
            if type(name) is np.int64:  name = (name,0)
            if name[0] == 1:
                alts = alts1
                ind_vars = ind_vars1
            if name[0] == 2:
                alts = alts2
                ind_vars = ind_vars2
            if name[0] == 3:
                alts = alts3
                ind_vars = ind_vars3
            if name[0] == 4:
                alts = alts4
                ind_vars = ind_vars4
            segment = segment.head(1)
            name_coeff = str(name[0])
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
            alternative_sample['high_inc_x_percent_high_inc'] = (alternative_sample.high_inc*alternative_sample.percent_high_income)
            alternative_sample['mid_inc_x_percent_mid_inc'] = (alternative_sample.mid_inc*alternative_sample.percent_mid_income)
            alternative_sample['low_inc_x_percent_low_inc'] = (alternative_sample.low_inc*alternative_sample.percent_low_income)
            alternative_sample['hhsize3plus_x_percent_hhsize3plus'] = (alternative_sample.hhsize3plus*alternative_sample.percent_hhsize3plus)
            alternative_sample['hhsize2_x_percent_hhsize2'] = (alternative_sample.hhsize2*alternative_sample.percent_hhsize2)
            #alternative_sample['same_dpt_as_previous'] = (alternative_sample.previous_dpt==alternative_sample.dept_id).astype('int32')
            alternative_sample['rail_stations_x_0car'] = (alternative_sample.ctrain9*(alternative_sample.cars==0)).astype('int32')
            alternative_sample['rail_stations_x_1car'] = (alternative_sample.ctrain9*(alternative_sample.cars==1)).astype('int32')
            alternative_sample['rail_stations_x_2pluscar'] = (alternative_sample.ctrain9*(alternative_sample.cars>1)).astype('int32')
            alternative_sample['subway_stations_x_0car'] = (alternative_sample.csubway9*(alternative_sample.cars==0)).astype('int32')
            alternative_sample['subway_stations_x_1car'] = (alternative_sample.csubway9*(alternative_sample.zgpgroup75==1)*(alternative_sample.cars==1)).astype('int32')
            alternative_sample['subway_stations_x_2pluscar'] = (alternative_sample.csubway9*(alternative_sample.zgpgroup75==1)*(alternative_sample.cars>1)).astype('int32')
            alternative_sample['tco_x_0car'] = (alternative_sample.tco*(alternative_sample.cars==0)).astype('int32')
            alternative_sample['tco_x_1car'] = (alternative_sample.tco*(alternative_sample.cars==1)).astype('int32')
            alternative_sample['tco_x_2pluscar'] = (alternative_sample.tco*(alternative_sample.cars>1)).astype('int32')
            alternative_sample['vpo_x_0car'] = (alternative_sample.vpo*(alternative_sample.cars==0)).astype('int32')
            alternative_sample['vpo_x_1car'] = (alternative_sample.vpo*(alternative_sample.cars==1)).astype('int32')
            alternative_sample['vpo_x_2pluscar'] = (alternative_sample.vpo*(alternative_sample.cars>1)).astype('int32')
            alternative_sample['perc_gardens_x_children'] = (alternative_sample.cpparc_jardin*alternative_sample.children).astype('int32')
            alternative_sample['perc_foreign_x_french'] = (alternative_sample.percent_foreigners*(alternative_sample.race_id==0))
            alternative_sample['perc_foreign_x_foreign'] = (alternative_sample.percent_foreigners*(alternative_sample.race_id==1))
            alternative_sample['perc_young_x_young'] = (alternative_sample.percent_young*alternative_sample.young)
            alternative_sample['perc_middle_age_x_middle_age'] = (alternative_sample.percent_middle_age*alternative_sample.middle_age)
            alternative_sample['perc_old_x_old'] = (alternative_sample.percent_old*alternative_sample.old)
            alternative_sample['perc_with_child_x_child_in_hh'] = (alternative_sample.with_child*alternative_sample.percent_with_child)
            alternative_sample['percent_middle_age_x_high_inc'] = (alternative_sample.high_inc*alternative_sample.percent_middle_age)
            alternative_sample['percent_old_x_low_inc'] = (alternative_sample.low_inc*alternative_sample.percent_old)
            alternative_sample['percent_middle_age_x_mid_inc'] = (alternative_sample.mid_inc*alternative_sample.percent_middle_age)
            alternative_sample['not_paris_subway_stations_x_1car'] = (alternative_sample.csubway9*(alternative_sample.zgpgroup75<>1)*(alternative_sample.cars==1)).astype('int32')
            alternative_sample['not_paris_subway_stations_x_2pluscar'] = (alternative_sample.csubway9*(alternative_sample.zgpgroup75<>1)*(alternative_sample.cars>1)).astype('int32')
            if int(name_coeff)==1:
                alternative_sample['ln_price1_x_low_income'] = (alternative_sample.ln_average_res_price1*alternative_sample.low_inc)
                alternative_sample['ln_price1_x_mid_income'] = (alternative_sample.ln_average_res_price1*alternative_sample.mid_inc)
                alternative_sample['ln_price1_x_high_income'] = (alternative_sample.ln_average_res_price1*alternative_sample.high_inc)
            if int(name_coeff)==2:
                alternative_sample['ln_price2_x_low_income'] = (alternative_sample.ln_average_res_price2*alternative_sample.low_inc)
                alternative_sample['ln_price2_x_mid_income'] = (alternative_sample.ln_average_res_price2*alternative_sample.mid_inc)
                alternative_sample['ln_price2_x_high_income'] = (alternative_sample.ln_average_res_price2*alternative_sample.high_inc)
            if int(name_coeff)==3:
                alternative_sample['ln_price3_x_low_income'] = (alternative_sample.ln_average_res_price3*alternative_sample.low_inc)
                alternative_sample['ln_price3_x_mid_income'] = (alternative_sample.ln_average_res_price3*alternative_sample.mid_inc)
                alternative_sample['ln_price3_x_high_income'] = (alternative_sample.ln_average_res_price3*alternative_sample.high_inc)
            if int(name_coeff)==4:
                alternative_sample['ln_price4_x_low_income'] = (alternative_sample.ln_average_res_price4*alternative_sample.low_inc)
                alternative_sample['ln_price4_x_mid_income'] = (alternative_sample.ln_average_res_price4*alternative_sample.mid_inc)
                alternative_sample['ln_price4_x_high_income'] = (alternative_sample.ln_average_res_price4*alternative_sample.high_inc)
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
     
        new_homes = pd.Series(np.ones(len(movers.index))*-1,index=movers.index)
        #mask = np.zeros(len(alternatives.index),dtype='bool')
        for name, segment in segments:
            if type(name) is np.int64:  name = (name,0)
            name_coeff = str(name[0])
            name = str(name)
            if int(name_coeff) == 1:
                p=pdf1['segment%s'%name].values
                mask = np.zeros(len(alts1.index),dtype='bool')
            if int(name_coeff) == 2:
                p=pdf2['segment%s'%name].values 
                mask = np.zeros(len(alts2.index),dtype='bool')
            if int(name_coeff) == 3:
                p=pdf3['segment%s'%name].values 
                mask = np.zeros(len(alts3.index),dtype='bool')
            if int(name_coeff) == 4:
                p=pdf4['segment%s'%name].values
                mask = np.zeros(len(alts4.index),dtype='bool')
            print "Assigning units to %d agents of segment %s" % (len(segment.index),name)
            #p=pdf['segment%s'%name].values
         
            def choose(p,mask,alternatives,segment,new_homes,minsize=None):
                p = copy.copy(p)
                p[mask] = 0 # already chosen
                #print "Choosing from %d nonzero alts" % np.count_nonzero(p)
                try: 
                  indexes = np.random.choice(len(alternatives.index),len(segment.index),replace=False,p=p/p.sum())
                except:
                  print "WARNING: not enough options to fit agents, will result in unplaced agents"
                  return mask,new_homes
                new_homes.ix[segment.index] = alternatives.index.values[indexes]
                mask[indexes] = 1
              
                return mask,new_homes
            if int(name_coeff) == 1:
                mask,new_homes = choose(p,mask,alts1,segment,new_homes)
            if int(name_coeff) == 2:
                mask,new_homes = choose(p,mask,alts2,segment,new_homes)
            if int(name_coeff) == 3:
                mask,new_homes = choose(p,mask,alts3,segment,new_homes)
            if int(name_coeff) == 4:
                mask,new_homes = choose(p,mask,alts4,segment,new_homes)
            
        build_cnts = new_homes.value_counts()  #num households place in each building
        print "Assigned %d agents to %d locations with %d unplaced" % (new_homes.size,build_cnts.size,build_cnts.get(-1,0))
        
        table = dset.households # need to go back to the whole dataset
        table[depvar].ix[new_homes.index] = new_homes.values.astype('int32')
        dset.store_attr(output_varname,year,copy.deepcopy(table[depvar]))
        
        
    #################     RDPLCM
        target_vacancy1 = .081
        target_vacancy2 = .081
        target_vacancy3 = .081
        target_vacancy4 = .081
        target_vacancy6 = .097
        target_vacancies = pd.Series([target_vacancy1,target_vacancy2,target_vacancy3,target_vacancy4,target_vacancy6],index=[1,2,3,4,6])
        households_by_btype = dset.households.groupby('btype_tenure').building_id.count()
        resunits_by_btype = dset.buildings[dset.buildings.building_type_id<5].groupby('building_type_id').residential_units.sum()
        vacant_resunits = resunits_by_btype - households_by_btype
        vacant_resunits = vacant_resunits[vacant_resunits.index.values<6]
        target_vacant_resunits = resunits_by_btype * target_vacancies
        diff_resunits = np.round(target_vacant_resunits - vacant_resunits)
        print 'Residential units by building type to construct:  '
        building_type_ids = []
        building_ids = []
        for idx_btype in diff_resunits[diff_resunits>0].index:
            building_type_id = idx_btype
            residential_units_to_build = int(diff_resunits[idx_btype])
            print building_type_id, residential_units_to_build
            building_type_ids += [building_type_id]*residential_units_to_build
            building_ids += [-1]*residential_units_to_build
        building_type_ids = np.array(building_type_ids)
        building_ids = np.array(building_ids)
        residential_unit_ids = np.arange(len(building_ids))+1
        residential_units = pd.DataFrame({'building_type_id':building_type_ids,'building_id':building_ids,'residential_unit_id':residential_unit_ids})
        residential_units = residential_units.set_index('residential_unit_id')
        output_csv, output_title, coeff_name, output_varname = ("paris-coeff-dplcm-%s.csv","PARIS DEVPROJECT LOCATION CHOICE MODELS (%s)","dp_location_%s","devproject_building_ids")
        year = sim_year
        choosers = residential_units
        depvar = 'building_id'
        movers = choosers[choosers[depvar]==-1]
        print "Total new agents and movers = %d" % len(movers.index)
        alternatives = dset.buildings[(dset.buildings.residential_units_capacity>0)]
        alternatives.residential_units_capacity[alternatives.zgpgroup75==1] = np.round(alternatives.residential_units_capacity[alternatives.zgpgroup75==1]*.6)
        empty_units = dset.buildings[(dset.buildings.residential_units_capacity>0)].residential_units_capacity.sub(dset.buildings.residential_units,fill_value=0)
        empty_units = empty_units[empty_units>0].order(ascending=False)
        empty_units[empty_units>2000] = 2000
        ind_vars1=['in_new_town','in_paris','in_paris_suburbs','distance_to_arterial','distance_to_highway','cd_chatelet','ln_land_area','csubway9','ctrain9',
                   'percent_high_income','percent_low_income','cnoise','cpbois','cpequipem_sante','cpparc_jardin','cpsport','cpeau','cpraft90','percent_hh_one_worker',
                   'percent_hh_twoplus_workers','cprbef15','percent_foreigners','percent_young','percent_old','percent_hhsize2', 'percent_hhsize3plus',
                   'employment_density','population_density','vpo',] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
        ind_vars2=['in_new_town','in_paris','in_paris_suburbs','distance_to_arterial','distance_to_highway','cd_chatelet','ln_land_area','csubway9','ctrain9',
                   'percent_high_income','percent_low_income','cnoise','cpbois','cpequipem_sante','cpparc_jardin','cpsport','cpeau','cpraft90','percent_hh_one_worker',
                   'percent_hh_twoplus_workers','cprbef15','percent_foreigners','percent_young','percent_old','percent_hhsize2', 'percent_hhsize3plus',
                   'employment_density','population_density','vpo',] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
        ind_vars3=['in_new_town','in_paris','in_paris_suburbs','distance_to_arterial','distance_to_highway','cd_chatelet','ln_land_area','csubway9','ctrain9',
                   'percent_high_income','percent_low_income','cnoise','cpbois','cpequipem_sante','cpparc_jardin','cpsport','cpeau','cpraft90','percent_hh_one_worker',
                   'percent_hh_twoplus_workers','cprbef15','percent_foreigners','percent_young','percent_old','percent_hhsize2', 'percent_hhsize3plus',
                   'employment_density','population_density','vpo',] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
        ind_vars4=['in_new_town','in_paris','in_paris_suburbs','distance_to_arterial','distance_to_highway','cd_chatelet','ln_land_area','csubway9','ctrain9',
                   'percent_high_income','percent_low_income','cnoise','cpbois','cpequipem_sante','cpparc_jardin','cpsport','cpeau','cpraft90','percent_hh_one_worker',
                   'percent_hh_threeplus_workers','cprbef15','percent_foreigners','percent_young','percent_old','percent_hhsize2', 'percent_hhsize3plus',
                   'employment_density','population_density','vpo',] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
        indvars_together = ind_vars1 + ind_vars2 + ind_vars3 + ind_vars4 + ['building_type_id','zone_id','zgp_id','dept_id','residential_units','residential_units_capacity','non_residential_sqft','non_residential_sqft_capacity']
        columns_to_keep = np.unique(indvars_together)
        alternatives = alternatives[list(columns_to_keep)]
        alternatives = alternatives.ix[np.repeat(empty_units.index,empty_units.values.astype('int'))]
        alts1 = alternatives[alternatives.building_type_id==1]
        alts2 = alternatives[alternatives.building_type_id==2]
        alts3 = alternatives[alternatives.building_type_id==3]
        alts4 = alternatives[alternatives.building_type_id==4]
        
        segments = movers.groupby(['building_type_id',])
        
        for name, segment in segments:
            if name == 1:
                alts = alts1
                ind_vars = ind_vars1
                pdf1 = pd.DataFrame(index=alts1.index) 
            if name == 2:
                alts = alts2
                ind_vars = ind_vars2
                pdf2 = pd.DataFrame(index=alts2.index) 
            if name == 3:
                alts = alts3
                ind_vars = ind_vars3
                pdf3 = pd.DataFrame(index=alts3.index) 
            if name == 4:
                alts = alts4
                ind_vars = ind_vars4
                pdf4 = pd.DataFrame(index=alts4.index) 
            
            segment = segment.head(1)
            name = str(name)
            tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
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
            est_data = pd.DataFrame(index=alternative_sample.index)
            for varname in ind_vars:
                est_data[varname] = alternative_sample[varname]
            est_data = est_data.fillna(0)
            data = est_data
            data = data.as_matrix()
            coeff = dset.load_coeff(tmp_coeffname)
            probs = interaction.mnl_simulate(data,coeff,numalts=SAMPLE_SIZE,returnprobs=1)
            if int(name) == 1:
                pdf1['segment%s'%name] = pd.Series(probs.flatten(),index=alts.index)  
            if int(name) == 2:
                pdf2['segment%s'%name] = pd.Series(probs.flatten(),index=alts.index) 
            if int(name) == 3:
                pdf3['segment%s'%name] = pd.Series(probs.flatten(),index=alts.index) 
            if int(name) == 4:
                pdf4['segment%s'%name] = pd.Series(probs.flatten(),index=alts.index) 
     
        new_homes = pd.Series(np.ones(len(movers.index))*-1,index=movers.index)
        #mask = np.zeros(len(alternatives.index),dtype='bool')
        for name, segment in segments:
            name = str(name)
            if int(name) == 1:
                p=pdf1['segment%s'%name].values
                mask = np.zeros(len(alts1.index),dtype='bool')
            if int(name) == 2:
                p=pdf2['segment%s'%name].values 
                mask = np.zeros(len(alts2.index),dtype='bool')
            if int(name) == 3:
                p=pdf3['segment%s'%name].values 
                mask = np.zeros(len(alts3.index),dtype='bool')
            if int(name) == 4:
                p=pdf4['segment%s'%name].values
                mask = np.zeros(len(alts4.index),dtype='bool')
            print "Assigning units to %d agents of segment %s" % (len(segment.index),name)
            #p=pdf['segment%s'%name].values
         
            def choose(p,mask,alternatives,segment,new_homes,minsize=None):
                p = copy.copy(p)
                if minsize is not None: p[alternatives.supply<minsize] = 0
                else: p[mask] = 0 # already chosen
    #             print "Choosing from %d nonzero alts" % np.count_nonzero(p)
                try: 
                  indexes = np.random.choice(len(alternatives.index),len(segment.index),replace=False,p=p/p.sum())
                except:
                  print "WARNING: not enough options to fit agents, will result in unplaced agents"
                  return mask,new_homes
                new_homes.ix[segment.index] = alternatives.index.values[indexes]
            
                if minsize is not None: alternatives["supply"].ix[alternatives.index.values[indexes]] -= minsize
                else: mask[indexes] = 1
              
                return mask,new_homes
            if int(name) == 1:
                mask,new_homes = choose(p,mask,alts1,segment,new_homes)
            if int(name) == 2:
                mask,new_homes = choose(p,mask,alts2,segment,new_homes)
            if int(name) == 3:
                mask,new_homes = choose(p,mask,alts3,segment,new_homes)
            if int(name) == 4:
                mask,new_homes = choose(p,mask,alts4,segment,new_homes)
            
        build_cnts = new_homes.value_counts()  #num resunits place in each building
        print "Assigned %d agents to %d locations with %d unplaced" % (new_homes.size,build_cnts.size,build_cnts.get(-1,0))
        
        table = residential_units # need to go back to the whole dataset *****************
        table[depvar].ix[new_homes.index] = new_homes.values.astype('int32')
        dset.store_attr(output_varname,year,copy.deepcopy(table[depvar]))
        new_res_construction_totals = residential_units.groupby('building_id').size()
        print 'Previous residential unit total:'
        print dset.buildings.residential_units.sum()
        dset.buildings.residential_units[np.in1d(dset.buildings.index,new_res_construction_totals.index)] = dset.buildings.residential_units[np.in1d(dset.buildings.index,new_res_construction_totals.index)] + new_res_construction_totals
        print 'Current residential unit total:'
        print dset.buildings.residential_units.sum()
        
        
    #################     NRDPLCM
        target_vacancy6 = .147
        employment_by_building = dset.establishments.groupby('building_id').employees.sum()
        bsqft_job = dset.buildings[dset.buildings.building_type_id==6].building_sqft_per_job
        occupied_nonres_sqft = employment_by_building*bsqft_job
        total_occupied_nonres_sqft = occupied_nonres_sqft.sum()
        total_nonres_sqft = dset.buildings.non_residential_sqft.sum()
        vacant_nonres_sqft = total_nonres_sqft - total_occupied_nonres_sqft
        target_vacant_nonres_sqft = total_nonres_sqft*target_vacancy6
        nonres_sqft_to_build = target_vacant_nonres_sqft - vacant_nonres_sqft
        print 'Non-residential sqft to construct:  '
        print nonres_sqft_to_build
        nonres_units_to_build = int(round(nonres_sqft_to_build/500.0))
        print 'Non-residential units to construct:  '
        print nonres_units_to_build
        building_type_ids = [6]*nonres_units_to_build
        nonres_building_ids = [-1]*nonres_units_to_build
        building_type_ids = np.array(building_type_ids)
        nonres_building_ids = np.array(nonres_building_ids)
        nonres_unit_ids = np.arange(len(nonres_building_ids))+1
        nonres_units = pd.DataFrame({'building_type_id':building_type_ids,'building_id':nonres_building_ids,'nonres_unit_id':nonres_unit_ids})
        nonres_units = nonres_units.set_index('nonres_unit_id')
        output_csv, output_title, coeff_name, output_varname = ("paris-coeff-dplcm-%s.csv","PARIS DEVPROJECT LOCATION CHOICE MODELS (%s)","dp_location_%s","devproject_building_ids")
        year = sim_year
        choosers = nonres_units
        depvar = 'building_id'
        movers = choosers[choosers[depvar]==-1]
        print "Total new agents and movers = %d" % len(movers.index)
        alternatives = dset.buildings[(dset.buildings.non_residential_sqft_capacity>0)*(dset.buildings.zgpgroup75!=1)]
        alternatives['nonres_units'] = alternatives.non_residential_sqft/500
        alternatives['nonres_units_capacity'] = alternatives.non_residential_sqft_capacity/500
        empty_units = alternatives.nonres_units_capacity.sub(alternatives.nonres_units,fill_value=0)
        empty_units = empty_units[empty_units>0].order(ascending=False)
        empty_units[empty_units>2500] = 2500
        ind_vars6=['in_la_defense','in_new_town','in_paris','in_paris_suburbs','distance_to_arterial','distance_to_highway','cd_chatelet','ln_land_area','csubway9','ctrain9',
                   'percent_high_income','percent_low_income','cnoise','cpraft90','percent_hh_one_worker','percent_hh_twoplus_workers','cprbef15','percent_foreigners','percent_young','percent_old',
                   'percent_hhsize2', 'percent_hhsize3plus','employment_density','population_density','vpo'
                   ] + ['zgpgroup21','zgpgroup22','zgpgroup23','zgpgroup24','zgpgroup25','zgpgroup26','zgpgroup47','zgpgroup48','zgpgroup49','zgpgroup50','zgpgroup75','zgpgroup99']
        indvars_together = ind_vars6 + ['building_type_id','zone_id','zgp_id','dept_id','residential_units','residential_units_capacity','non_residential_sqft','non_residential_sqft_capacity']
        columns_to_keep = np.unique(indvars_together)
        alternatives = alternatives[list(columns_to_keep)]
        alts6 = alternatives.ix[np.repeat(empty_units.index,empty_units.values.astype('int'))]
        
        segments = movers.groupby(['building_type_id',])
        
        for name, segment in segments:
            if name == 6:
                alts = alts6
                ind_vars = ind_vars6
                pdf6 = pd.DataFrame(index=alts.index) 
            
            segment = segment.head(1)
            name = str(name)
            tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
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
            est_data = pd.DataFrame(index=alternative_sample.index)
            for varname in ind_vars:
                est_data[varname] = alternative_sample[varname]
            est_data = est_data.fillna(0)
            data = est_data
            data = data.as_matrix()
            coeff = dset.load_coeff(tmp_coeffname)
            probs = interaction.mnl_simulate(data,coeff,numalts=SAMPLE_SIZE,returnprobs=1)
            if int(name) == 6:
                pdf6['segment%s'%name] = pd.Series(probs.flatten(),index=alts.index) 
        new_homes = pd.Series(np.ones(len(movers.index))*-1,index=movers.index)
        for name, segment in segments:
            name = str(name)
            if int(name) == 6:
                p=pdf6['segment%s'%name].values
                mask = np.zeros(len(alts6.index),dtype='bool')
            print "Assigning units to %d agents of segment %s" % (len(segment.index),name)
         
            def choose(p,mask,alternatives,segment,new_homes,minsize=None):
                p = copy.copy(p)
                if minsize is not None: p[alternatives.supply<minsize] = 0
                else: p[mask] = 0 # already chosen
                #print "Choosing from %d nonzero alts" % np.count_nonzero(p)
        
                try: 
                  indexes = np.random.choice(len(alternatives.index),len(segment.index),replace=False,p=p/p.sum())
                except:
                  print "WARNING: not enough options to fit agents, will result in unplaced agents"
                  return mask,new_homes
                new_homes.ix[segment.index] = alternatives.index.values[indexes]
            
                if minsize is not None: alternatives["supply"].ix[alternatives.index.values[indexes]] -= minsize
                else: mask[indexes] = 1
              
                return mask,new_homes
            if int(name) == 6:
                mask,new_homes = choose(p,mask,alts6,segment,new_homes)
            
        build_cnts = new_homes.value_counts()  #num resunits place in each building
        print "Assigned %d agents to %d locations with %d unplaced" % (new_homes.size,build_cnts.size,build_cnts.get(-1,0))
        
        table = nonres_units # need to go back to the whole dataset *****************
        table[depvar].ix[new_homes.index] = new_homes.values.astype('int32')
        dset.store_attr('nr'+output_varname,year,copy.deepcopy(table[depvar]))
        new_nonres_construction_totals = table.groupby('building_id').size()*500
        print 'Previous non-residential sqft total:'
        print dset.buildings.non_residential_sqft.sum()
        dset.buildings.non_residential_sqft[np.in1d(dset.buildings.index,new_nonres_construction_totals.index)] = dset.buildings.non_residential_sqft[np.in1d(dset.buildings.index,new_nonres_construction_totals.index)] + new_nonres_construction_totals
        print 'Current non-residential sqft total:'
        print dset.buildings.non_residential_sqft.sum()
    
    
    #################     REPM
        year = sim_year
        buildings = dset.fetch('buildings')
        output_csv, output_title, coeff_name, output_varname = ["paris-coeff-hedonic.csv","PARIS HEDONIC MODEL","price_%s","price"]
        ind_vars1 = ['in_paris_suburbs','percent_low_income','cpraft90','cprbef15','percent_old','bati',
                     'population_density','tco',]
        ind_vars2 = ['in_paris','in_paris_suburbs','csubway9','percent_low_income','cpraft90','cprbef15','percent_old','bati',
                     'population_density','tco',]
        ind_vars3 = ['in_paris_suburbs','csubway9','percent_low_income','cpraft90','cprbef15','percent_old',
                     'population_density','tco',]
        ind_vars4 = ['in_paris','in_paris_suburbs','csubway9','percent_low_income','cpraft90','cprbef15','percent_old','bati',
                     'population_density','tco',]
        ind_vars6 = ['in_paris','in_la_defense','in_new_town','in_paris_suburbs','csubway9','percent_low_income','cpraft90','cprbef15','bati',
                     'employment_density','tax_on_professionals','tco',]
        simrents = []
        segments = buildings.groupby('building_type_id')
        for name, segment in segments:
            if name == 1:
                indvars = ind_vars1
            if name == 2:
                indvars = ind_vars2
            if name == 3:
                indvars = ind_vars3
            if name == 4:
                indvars = ind_vars4
            if name == 6:
                indvars = ind_vars6
            est_data = pd.DataFrame(index=segment.index)
            for varname in indvars:
                est_data[varname] = segment[varname]
            est_data = est_data.fillna(0)
            est_data = sm.add_constant(est_data,prepend=False)
            tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
            print "Generating rents on %d buildings" % (est_data.shape[0])
            vec = dset.load_coeff(tmp_coeffname)
            vec = np.reshape(vec,(vec.size,1))
            rents = est_data.dot(vec).astype('f4')
            rents = rents.apply(np.exp)
            simrents.append(rents[rents.columns[0]])
            
        simrents = pd.concat(simrents)
        dset.buildings[output_varname] = simrents.reindex(dset.buildings.index)
        dset.store_attr(output_varname,year,simrents)
        
    #######ANNUAL SUMMARY
        b = dset.fetch('buildings')
        e = dset.fetch('establishments')
        hh = dset.fetch('households')
        summary['employment'].append(e[e.building_id>0].employees.sum())
        summary['households'].append(len(hh[hh.building_id>0].building_id))
        summary['non_residential_sqft'].append(b.non_residential_sqft.sum())
        summary['residential_units'].append(b.residential_units.sum())
        summary['price'].append(b.price.mean())
        
        ##End-of-iteration calibration update
        if sim_year == last_year:
            print summary
            hh['zone_id'] = b.zone_id[hh.building_id].values
            e['zone_id'] = b.zone_id[e.building_id].values
            z['total_households'] = hh.groupby('zone_id').building_id.count()
            z['total_residential_units'] = b.groupby('zone_id').residential_units.sum()
            z['total_nonresidential_sqft'] = b.groupby('zone_id').non_residential_sqft.sum()
            z['total_employment'] = e.groupby('zone_id').employees.sum()
            z['total_persons'] = hh.groupby('zone_id').persons.sum()
            sim_pop = z.groupby('zgpgroup_id').total_persons.sum()
            sim_emp = z.groupby('zgpgroup_id').total_employment.sum()
            sim_hh = z.groupby('zgpgroup_id').total_households.sum()
            sim_ru = z.groupby('zgpgroup_id').total_residential_units.sum()
            sim_nr = z.groupby('zgpgroup_id').total_nonresidential_sqft.sum()
            
            emp_diff_zone = z.total_employment - base_emp_zone
            pop_diff_zone = z.total_persons - base_pop_zone
            zone_diffs = pd.DataFrame({'emp_diff':emp_diff_zone,'pop_diff':pop_diff_zone})
            
            pop_diff = sim_pop - base_pop
            emp_diff = sim_emp - base_emp
            hh_diff = sim_hh - base_hh
            ru_diff = sim_ru - base_ru
            nr_diff = sim_nr - base_nr
            print 'Employment in 2035, by ZGPGroup'
            print sim_emp
            print 'Households in 2035, by ZGPGroup'
            print sim_hh
            print 'Population in 2035, by ZGPGroup'
            print sim_pop
            print 'Residential units in 2035, by ZGPGroup'
            print sim_ru
            print 'Non-residential sqft in 2035 by ZGPGroup'
            print sim_nr
            print 'Employment growth 1999-2035, by ZGPGroup'
            print emp_diff
            print 'Household growth 1999-2035, by ZGPGroup'
            print hh_diff
            print 'Population growth 1999-2035, by ZGPGroup'
            print pop_diff
            print 'Residential unit growth 1999-2035, by ZGPGroup'
            print ru_diff
            print 'Non-residential sqft growth 1999-2035 by ZGPGroup'
            print nr_diff
            
            prop_growth_emp = emp_diff*1.0/emp_diff.sum()
            prop_growth_hh = hh_diff*1.0/hh_diff.sum()
            prop_growth_pop = pop_diff*1.0/pop_diff.sum()
            prop_growth_ru = ru_diff*1.0/ru_diff.sum()
            prop_growth_nr = nr_diff*1.0/nr_diff.sum()
            perc_growth_hh = (hh_diff)*100.0/base_hh
            perc_growth_pop = (pop_diff)*100.0/base_pop
            perc_growth_emp = (emp_diff)*100.0/base_emp
            perc_growth_ru = (ru_diff)*100.0/base_ru
            perc_growth_nr = (nr_diff)*100.0/base_nr
            print 'Proportion of employment growth captured by ZGPGroup'
            print prop_growth_emp
            print 'Proportion of household growth captured by ZGPGroup'
            print prop_growth_hh
            print 'Proportion of population growth captured by ZGPGroup'
            print prop_growth_pop
            print 'Proportion of residential unit growth captured by ZGPGroup'
            print prop_growth_ru
            print 'Proportion of non-residential sqft growth captured by ZGPGroup'
            print prop_growth_nr
            
            print 'Percent employment growth, ZGPGroup'
            print perc_growth_emp
            print 'Perrcent household growth, ZGPGroup'
            print perc_growth_hh
            print 'Percent population growth, ZGPGroup'
            print perc_growth_pop
            print 'Percent residential unit growth, ZGPGroup'
            print perc_growth_ru
            print 'Percent non-residential sqft growth, ZGPGroup'
            print perc_growth_nr
            
            population = pd.DataFrame({'simulated_amount':sim_pop,'difference':pop_diff,'proportion_captured':prop_growth_pop})
            employment = pd.DataFrame({'simulated_amount':sim_emp,'difference':emp_diff,'proportion_captured':prop_growth_emp})
            household = pd.DataFrame({'simulated_amount':sim_hh,'difference':hh_diff,'proportion_captured':prop_growth_hh})
            residential_unit = pd.DataFrame({'simulated_amount':sim_ru,'difference':ru_diff,'proportion_captured':prop_growth_ru})
            nonres_sqft = pd.DataFrame({'simulated_amount':sim_nr,'difference':nr_diff,'proportion_captured':prop_growth_nr})
            
            population.to_csv(output_dir + '\\' + scenario+ title + '_population.csv')
            employment.to_csv(output_dir + '\\' + scenario+ title + '_employment.csv')
            household.to_csv(output_dir + '\\' + scenario+ title + '_households.csv')
            residential_unit.to_csv(output_dir + '\\' + scenario+ title + '_residential_units.csv')
            nonres_sqft.to_csv(output_dir + '\\' + scenario+ title + '_nonresidential_sqft.csv')
            
            zone_diffs.to_csv(output_dir + '\\' + scenario + title + '_zone_diffs.csv')
            
            estabs_base = dset.store.establishments
            b = dset.fetch('buildings')
            estabs_base['dept_id'] = b.dept_id[estabs_base.building_id].values
            estabs_base_by_dept = estabs_base.groupby('dept_id').employees.sum()
            estabs_sim = dset.fetch('establishments')
            estabs_sim['dept_id'] = b.dept_id[estabs_sim.building_id].values
            estabs_sim_by_dept = estabs_sim.groupby('dept_id').employees.sum()
            print 'Dept Growth Rate, Employment'
            print (estabs_sim_by_dept - estabs_base_by_dept)*1.0/estabs_base_by_dept 
            
            hh_base = dset.store.households
            b = dset.fetch('buildings')
            hh_base['dept_id'] = b.dept_id[hh_base.building_id].values
            hh_base_by_dept = hh_base.groupby('dept_id').building_id.count()
            hh_sim = dset.fetch('households')
            hh_sim['dept_id'] = b.dept_id[hh_sim.building_id].values
            hh_sim_by_dept = hh_sim.groupby('dept_id').building_id.count()
            print 'Dept Growth Rate, Households'
            print (hh_sim_by_dept - hh_base_by_dept)*1.0/hh_base_by_dept
            
            z['resunits_base'] = dset.store.buildings.groupby('zone_id').residential_units.sum()
            resunits_base_by_dept = z.groupby('dept_id').resunits_base.sum()
            b = dset.fetch('buildings')
            resunits_sim_by_dept = b.groupby('dept_id').residential_units.sum()
            print 'Dept Growth Rate, Residential Units'
            print (resunits_sim_by_dept - resunits_base_by_dept)*1.0/resunits_base_by_dept
        
            z['nonres_sqft_base'] = dset.store.buildings.groupby('zone_id').non_residential_sqft.sum()
            nonres_sqft_base_by_dept = z.groupby('dept_id').nonres_sqft_base.sum()
            b = dset.fetch('buildings')
            nonres_sqft_sim_by_dept = b.groupby('dept_id').non_residential_sqft.sum()
            print 'Dept Growth Rate, Non-residential sqft'
            print (nonres_sqft_sim_by_dept - nonres_sqft_base_by_dept)*1.0/nonres_sqft_base_by_dept
            
            sim_store_path = os.path.join(output_dir,'sim_store.h5')
            sim_store = pd.HDFStore(sim_store_path)
            sim_store[scenario+ title + '_households'] = hh
            sim_store[scenario+ title + '_establishments'] = e
            sim_store[scenario+ title + '_buildings'] = b
            sim_store.close()
            
    elapsed = time.time() - seconds_start
    print "TOTAL elapsed time: " + str(elapsed) + " seconds."
    
print 'Done running all scenarios.'