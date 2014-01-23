# Opus/UrbanSim urban simulation software.
# Copyright (C) 2010-2011 University of California, Berkeley, 2005-2009 University of Washington
# See opus_core/LICENSE 

from opus_core.model import Model
from opus_core.logger import logger
from opus_core.session_configuration import SessionConfiguration
import numpy as np
import pandas as pd

class Urbansim2(Model):
    """Runs an UrbanSim2 scenario
    """
    model_name = "UrbanSim2"
    
    def __init__(self,scenario='Base Scenario'):
        self.scenario = scenario

    def run(self, name=None, export_buildings_to_urbancanvas=False, base_year=2010, forecast_year=None, fixed_seed=True, random_seed=1, export_indicators=True, indicator_output_directory='C:/opus/data/drcog2/runs', core_components_to_run=None, household_transition=None,household_relocation=None,employment_transition=None, elcm_configuration=None, developer_configuration=None):
        """Runs an UrbanSim2 scenario 
        """
        logger.log_status('Starting UrbanSim2 run.')
        
########ESTIMATION
        logger.log_status('Model estimation...')
        
        import numpy as np, pandas as pd, os, statsmodels.api as sm
        import synthicity.urbansim.interaction as interaction
        from synthicity.utils import misc
        import dataset, copy, time, math
        dset = dataset.DRCOGDataset(os.path.join(misc.data_dir(),'drcog.h5'))
        np.random.seed(1)

        #VARIABLE LIBRARY
        #parcel
        p = dset.fetch('parcels')
        p['in_denver'] = (p.county_id==8031).astype('int32')
        #building
        b = dset.fetch('buildings',building_sqft_per_job_table=elcm_configuration['building_sqft_per_job_table'],bsqft_job_scaling=elcm_configuration['scaling_factor'])
        b['zone_id'] = p.zone_id[b.parcel_id].values
        #b['btype_dplcm'] = 1*(b.building_type_id==2) + 2*(b.building_type_id==3) + 3*(b.building_type_id==20) + 4*(b.building_type_id==24) + 6*np.invert(np.in1d(b.building_type_id,[2,3,20,24]))
        b['btype_hlcm'] = 1*(b.building_type_id==2) + 2*(b.building_type_id==3) + 3*(b.building_type_id==20) + 4*np.invert(np.in1d(b.building_type_id,[2,3,20]))
        #household
        hh_estim = dset.fetch('households_for_estimation')
        hh = dset.fetch('households')
        for table in [hh_estim, hh]:
            choosers = table
            choosers['building_type_id'] = b.building_type_id[choosers.building_id].values
            choosers['btype'] = 1*(choosers.building_type_id==2) + 2*(choosers.building_type_id==3) + 3*(choosers.building_type_id==20) + 4*np.invert(np.in1d(choosers.building_type_id,[2,3,20]))
        #establishment
        e = dset.fetch('establishments')
        e['sector_id_six'] = 1*(e.sector_id==61) + 2*(e.sector_id==71) + 3*np.in1d(e.sector_id,[11,21,22,23,31,32,33,42,48,49]) + 4*np.in1d(e.sector_id,[7221,7222,7224]) + 5*np.in1d(e.sector_id,[44,45,7211,7212,7213,7223]) + 6*np.in1d(e.sector_id,[51,52,53,54,55,56,62,81,92])
        #zone
        z = dset.fetch('zones')
        z['residential_units_zone'] = b.groupby('zone_id').residential_units.sum()
        z['non_residential_sqft_zone'] = b.groupby('zone_id').non_residential_sqft.sum()
        z['percent_sf'] = b[b.btype_hlcm==3].groupby('zone_id').residential_units.sum()*100.0/(b.groupby('zone_id').residential_units.sum())
        #merge parcels with zones
        pz = pd.merge(p,z,left_on='zone_id',right_index=True)
        #merge buildings with parcels/zones
        bpz = pd.merge(b,pz,left_on='parcel_id',right_index=True)
        bpz['residential_units_capacity'] = bpz.parcel_sqft/1500 - bpz.residential_units
        bpz.residential_units_capacity[bpz.residential_units_capacity<0] = 0
        dset.d['buildings'] = bpz

        ###########################
        if core_components_to_run['HLCM']:
            print 'HLCM'
            ##HCLM ESTIMATION

            depvar = 'building_id'
            SAMPLE_SIZE=100 ##for alts
            max_segment_size = 1200 ##for obs
            estimation_table = 'households_for_estimation'
            output_csv, output_title, coeff_name, output_varname = ("drcog-coeff-hlcm-%s.csv","DRCOG HOUSEHOLD LOCATION CHOICE MODELS (%s)","hh_location_%s","household_building_ids")
            agents_groupby_txt = ['btype',]

            choosers = dset.fetch(estimation_table)
            #Filter on agents

            #ALTS
            alternatives = dset.buildings  #alternatives = dset.buildings[(dset.buildings.residential_units>0)]
            ##We have 4 markets-  1. apartment, 2. condo, 3. single family, 4. townhome and other
            alts1 = alternatives[alternatives.btype_hlcm==1]
            alts2 = alternatives[alternatives.btype_hlcm==2]
            alts3 = alternatives[alternatives.btype_hlcm==3]
            alts4 = alternatives[alternatives.btype_hlcm==4]
            ind_vars1=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_residential','in_denver']
            ind_vars2=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_residential','in_denver']
            ind_vars3=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_residential','in_denver']
            ind_vars4=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_residential','in_denver']

            segments = choosers.groupby(agents_groupby_txt)
            for name, segment in segments:
                if name == 1:
                    alts = alts1
                    ind_vars = ind_vars1
                if name == 2:
                    alts = alts2
                    ind_vars = ind_vars2
                if name == 3:
                    alts = alts3
                    ind_vars = ind_vars3
                if name == 4:
                    alts = alts4
                    ind_vars = ind_vars4
                name = str(name)
                tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
                if len(segment[depvar]) > max_segment_size: #reduce size of segment if too big so things don't bog down
                    segment = segment.ix[np.random.choice(segment.index, max_segment_size,replace=False)]
                #,weight_var='residential_units')
                sample, alternative_sample, est_params = interaction.mnl_interaction_dataset(segment,alts,SAMPLE_SIZE,chosenalts=segment[depvar])
                ##Interaction variables defined here
                print "Estimating parameters for segment = %s, size = %d" % (name, len(segment.index)) 
                if len(segment.index) > 50:
                    est_data = pd.DataFrame(index=alternative_sample.index)
                    for varname in ind_vars:
                        est_data[varname] = alternative_sample[varname]
                    est_data = est_data.fillna(0)
                    data = est_data.as_matrix()
                    try:
                        fit, results = interaction.estimate(data, est_params, SAMPLE_SIZE)
                        fnames = interaction.add_fnames(ind_vars,est_params)
                        print misc.resultstotable(fnames,results)
                        misc.resultstocsv(fit,fnames,results,tmp_outcsv,tblname=tmp_outtitle)
                        dset.store_coeff(tmp_coeffname,zip(*results)[0],fnames)
                    except:
                        print 'SINGULAR MATRIX OR OTHER DATA/ESTIMATION PROBLEM'
                else:
                    print 'SAMPLE SIZE TOO SMALL'
                
        ########################
        if core_components_to_run['ELCM']:
            print 'ELCM'
            #ELCM ESTIMATION

            depvar = 'building_id'
            SAMPLE_SIZE=100 ##for alts
            max_segment_size = 1200 ##for obs
            estimation_table = 'establishments'
            output_csv, output_title, coeff_name, output_varname = ("drcog-coeff-elcm-%s.csv","DRCOG EMPLOYMENT LOCATION CHOICE MODELS (%s)","emp_location_%s","establishment_building_ids")
            agents_groupby_txt = ['sector_id_six',]

            choosers = dset.fetch(estimation_table)
            #Filter on agents
            choosers = choosers[(choosers.building_id>0)*(choosers.home_based_status==0)]
            #ALTS
            alts = dset.buildings  #alts = dset.buildings[(dset.buildings.non_residential_sqft>0)]
            ind_vars1=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
            ind_vars2=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
            ind_vars3=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
            ind_vars4=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
            ind_vars5=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
            ind_vars6=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]

            segments = choosers.groupby(agents_groupby_txt)
            for name, segment in segments:
                print name
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
                name = str(name)
                tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
                if len(segment[depvar]) > max_segment_size: #reduce size of segment if too big so things don't bog down
                    segment = segment.ix[np.random.choice(segment.index, max_segment_size,replace=False)]
                #,weight_var='non_residential_sqft')
                sample, alternative_sample, est_params = interaction.mnl_interaction_dataset(segment,alts,SAMPLE_SIZE,chosenalts=segment[depvar])
                ##Interaction variables defined here
                print "Estimating parameters for segment = %s, size = %d" % (name, len(segment.index)) 
                if len(segment.index) > 50:
                    est_data = pd.DataFrame(index=alternative_sample.index)
                    for varname in ind_vars:
                        est_data[varname] = alternative_sample[varname]
                    est_data = est_data.fillna(0)
                    data = est_data.as_matrix()
                    try:
                        fit, results = interaction.estimate(data, est_params, SAMPLE_SIZE)
                        fnames = interaction.add_fnames(ind_vars,est_params)
                        print misc.resultstotable(fnames,results)
                        misc.resultstocsv(fit,fnames,results,tmp_outcsv,tblname=tmp_outtitle)
                        dset.store_coeff(tmp_coeffname,zip(*results)[0],fnames)
                    except:
                        print 'SINGULAR MATRIX OR OTHER DATA/ESTIMATION PROBLEM'
                else:
                    print 'SAMPLE SIZE TOO SMALL'
                

        ######################
        if core_components_to_run['Developer']:
            print 'RDPLCM'
            ##RDPLCM ESTIMATION

            depvar = 'parcel_id'
            SAMPLE_SIZE=100 ##for alts
            max_segment_size = 1200 ##for obs
            estimation_table = 'buildings'
            output_csv, output_title, coeff_name, output_varname = ("drcog-coeff-rdplcm-%s.csv","DRCOG RES DEVPROJECT LOCATION CHOICE MODELS (%s)","rdp_location_%s","resbuilding_parcel_ids")
            agents_groupby_txt = ['building_type_id',]

            choosers = dset.store[estimation_table]
            #Filter on agents
            choosers = choosers[(np.in1d(choosers.building_type_id,[2,3,20,24]))*(choosers.year_built>2002)]
            #ALTS
            alternatives = dset.parcels  #alternatives = dset.buildings[(dset.buildings.residential_units>0)]
            ##We have 4 markets-  1. apartment, 2. condo, 3. single family, 4. townhome and other
            # alts1 = alternatives[alternatives.btype_hlcm==1]
            # alts2 = alternatives[alternatives.btype_hlcm==2]
            # alts3 = alternatives[alternatives.btype_hlcm==3]
            # alts4 = alternatives[alternatives.btype_hlcm==4]
            ##Need to break alternatives down by allowable use during simulation
            alts1 = alternatives
            alts2 = alternatives
            alts3 = alternatives
            alts4 = alternatives
            ind_vars1=['parcel_sqft','dist_bus','dist_rail','land_value']
            ind_vars2=['parcel_sqft','dist_bus','dist_rail','land_value']
            ind_vars3=['parcel_sqft','dist_bus','dist_rail','land_value']
            ind_vars4=['parcel_sqft','dist_bus','dist_rail','land_value']

            segments = choosers.groupby(agents_groupby_txt)
            for name, segment in segments:
                if name == 2:
                    alts = alts1
                    ind_vars = ind_vars1
                if name == 3:
                    alts = alts2
                    ind_vars = ind_vars2
                if name == 20:
                    alts = alts3
                    ind_vars = ind_vars3
                if name == 24:
                    alts = alts4
                    ind_vars = ind_vars4
                name = str(name)
                tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
                if len(segment[depvar]) > max_segment_size: #reduce size of segment if too big so things don't bog down
                    segment = segment.ix[np.random.choice(segment.index, max_segment_size,replace=False)]
                #,weight_var='parcel_sqft')
                sample, alternative_sample, est_params = interaction.mnl_interaction_dataset(segment,alts,SAMPLE_SIZE,chosenalts=segment[depvar])
                ##Interaction variables defined here
                print "Estimating parameters for segment = %s, size = %d" % (name, len(segment.index)) 
                if len(segment.index) > 50:
                    est_data = pd.DataFrame(index=alternative_sample.index)
                    for varname in ind_vars:
                        est_data[varname] = alternative_sample[varname]
                    est_data = est_data.fillna(0)
                    data = est_data.as_matrix()
                    try:
                        fit, results = interaction.estimate(data, est_params, SAMPLE_SIZE)
                        fnames = interaction.add_fnames(ind_vars,est_params)
                        print misc.resultstotable(fnames,results)
                        misc.resultstocsv(fit,fnames,results,tmp_outcsv,tblname=tmp_outtitle)
                        dset.store_coeff(tmp_coeffname,zip(*results)[0],fnames)
                    except:
                        print 'SINGULAR MATRIX OR OTHER DATA/ESTIMATION PROBLEM'
                else:
                    print 'SAMPLE SIZE TOO SMALL'


            ###########################
            print 'NRDPLCM'
            ##NRDPLCM ESTIMATION

            depvar = 'parcel_id'
            SAMPLE_SIZE=100 ##for alts
            max_segment_size = 1200 ##for obs
            estimation_table = 'buildings'
            output_csv, output_title, coeff_name, output_varname = ("drcog-coeff-nrdplcm-%s.csv","DRCOG NONRES DEVPROJECT LOCATION CHOICE MODELS (%s)","nrdp_location_%s","nonresbuilding_parcel_ids")
            agents_groupby_txt = ['building_type_id',]

            choosers = dset.store[estimation_table]
            #Filter on agents
            ##Commercial
            choosers = choosers[(np.in1d(choosers.building_type_id,[5,8,11,16,17,18,21,23,9,22]))*(choosers.year_built>2000)] ##9 and 22 are industrial and should have different allowable parcels
            #ALTS
            alternatives = dset.parcels  #alternatives = dset.buildings[(dset.buildings.residential_units>0)]
            ##We have 4 markets-  1. apartment, 2. condo, 3. single family, 4. townhome and other
            # alts1 = alternatives[alternatives.btype_hlcm==1]
            # alts2 = alternatives[alternatives.btype_hlcm==2]
            # alts3 = alternatives[alternatives.btype_hlcm==3]
            # alts4 = alternatives[alternatives.btype_hlcm==4]
            ##Need to break alternatives down by allowable use during simulation
            alts1 = alternatives
            alts2 = alternatives
            alts3 = alternatives
            alts4 = alternatives
            alts5 = alternatives
            alts6 = alternatives
            alts7 = alternatives
            alts8 = alternatives
            alts9 = alternatives
            alts10 = alternatives
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

            segments = choosers.groupby(agents_groupby_txt)
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
                name = str(name)
                tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
                if len(segment[depvar]) > max_segment_size: #reduce size of segment if too big so things don't bog down
                    segment = segment.ix[np.random.choice(segment.index, max_segment_size,replace=False)]
                #,weight_var='parcel_sqft')
                sample, alternative_sample, est_params = interaction.mnl_interaction_dataset(segment,alts,SAMPLE_SIZE,chosenalts=segment[depvar])
                ##Interaction variables defined here
                print "Estimating parameters for segment = %s, size = %d" % (name, len(segment.index)) 
                if len(segment.index) > 50:
                    est_data = pd.DataFrame(index=alternative_sample.index)
                    for varname in ind_vars:
                        est_data[varname] = alternative_sample[varname]
                    est_data = est_data.fillna(0)
                    data = est_data.as_matrix()
                    try:
                        fit, results = interaction.estimate(data, est_params, SAMPLE_SIZE)
                        fnames = interaction.add_fnames(ind_vars,est_params)
                        print misc.resultstotable(fnames,results)
                        misc.resultstocsv(fit,fnames,results,tmp_outcsv,tblname=tmp_outtitle)
                        dset.store_coeff(tmp_coeffname,zip(*results)[0],fnames)
                    except:
                        print 'SINGULAR MATRIX OR OTHER DATA/ESTIMATION PROBLEM'
                else:
                    print 'SAMPLE SIZE TOO SMALL'
                
                
                
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
            
            for sim_year in range(first_year,last_year+1):
                print 'Simulating year ' + str(sim_year)

            #####Variable calculations
                #parcel
                p = dset.fetch('parcels')
                p['in_denver'] = (p.county_id==8031).astype('int32')
                #building
                b = dset.fetch('buildings',building_sqft_per_job_table=elcm_configuration['building_sqft_per_job_table'],bsqft_job_scaling=elcm_configuration['scaling_factor'])
                b = b[['building_type_id','improvement_value','land_area','non_residential_sqft','parcel_id','residential_units','sqft_per_unit','stories','tax_exempt','year_built','bldg_sq_ft','unit_price_non_residential','unit_price_residential','building_sqft_per_job','non_residential_units','base_year_jobs','all_units']]
                b['zone_id'] = p.zone_id[b.parcel_id].values
                b['county_id'] = p.county_id[b.parcel_id].values
                #b['btype_dplcm'] = 1*(b.building_type_id==2) + 2*(b.building_type_id==3) + 3*(b.building_type_id==20) + 4*(b.building_type_id==24) + 6*np.invert(np.in1d(b.building_type_id,[2,3,20,24]))
                b['btype_hlcm'] = 1*(b.building_type_id==2) + 2*(b.building_type_id==3) + 3*(b.building_type_id==20) + 4*np.invert(np.in1d(b.building_type_id,[2,3,20]))
                #household
                hh_estim = dset.fetch('households_for_estimation')
                hh = dset.fetch('households')
                for table in [hh_estim, hh]:
                    choosers = table
                    choosers['building_type_id'] = b.building_type_id[choosers.building_id].values
                    choosers['county_id'] = b.county_id[choosers.building_id].values
                    choosers['btype'] = 1*(choosers.building_type_id==2) + 2*(choosers.building_type_id==3) + 3*(choosers.building_type_id==20) + 4*np.invert(np.in1d(choosers.building_type_id,[2,3,20]))
                #establishment
                e = dset.fetch('establishments')
                e['county_id'] = b.county_id[e.building_id].values
                e['sector_id_six'] = 1*(e.sector_id==61) + 2*(e.sector_id==71) + 3*np.in1d(e.sector_id,[11,21,22,23,31,32,33,42,48,49]) + 4*np.in1d(e.sector_id,[7221,7222,7224]) + 5*np.in1d(e.sector_id,[44,45,7211,7212,7213,7223]) + 6*np.in1d(e.sector_id,[51,52,53,54,55,56,62,81,92])
                #zone
                z = dset.fetch('zones')
                z['residential_units_zone'] = b.groupby('zone_id').residential_units.sum()
                z['non_residential_sqft_zone'] = b.groupby('zone_id').non_residential_sqft.sum()
                z['percent_sf'] = b[b.btype_hlcm==3].groupby('zone_id').residential_units.sum()*100.0/(b.groupby('zone_id').residential_units.sum())
                #merge parcels with zones
                pz = pd.merge(p,z,left_on='zone_id',right_index=True)
                #merge buildings with parcels/zones
                bpz = pd.merge(b,pz,left_on='parcel_id',right_index=True)
                bpz['residential_units_capacity'] = bpz.parcel_sqft/1500 - bpz.residential_units
                bpz.residential_units_capacity[bpz.residential_units_capacity<0] = 0
                dset.d['buildings'] = bpz
                        
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

        ############     ELCM SIMULATION
                if core_components_to_run['ELCM']:
                    depvar = 'building_id'
                    simulation_table = 'establishments'
                    output_csv, output_title, coeff_name, output_varname = ("drcog-coeff-elcm-%s.csv","DRCOG EMPLOYMENT LOCATION CHOICE MODELS (%s)","emp_location_%s","establishment_building_ids")
                    agents_groupby_txt = ['sector_id_six',]
                    
                    output_csv, output_title, coeff_name, output_varname = ("drcog-coeff-elcm-%s.csv","DRCOG EMPLOYMENT LOCATION CHOICE MODELS (%s)","emp_location_%s","establishment_building_ids")
                    dset.establishments['home_based_status']=0
                    
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
                    
                    ind_vars1=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
                    ind_vars2=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
                    ind_vars3=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
                    ind_vars4=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
                    ind_vars5=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
                    ind_vars6=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
                    
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
                    depvar = 'building_id'
                    simulation_table = 'households'
                    output_csv, output_title, coeff_name, output_varname = ("drcog-coeff-hlcm-%s.csv","DRCOG HOUSEHOLD LOCATION CHOICE MODELS (%s)","hh_location_%s","household_building_ids")
                    agents_groupby_txt = ['btype',]
                    
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
                    alts1 = alternatives[alternatives.btype_hlcm==1]
                    alts2 = alternatives[alternatives.btype_hlcm==2]
                    alts3 = alternatives[alternatives.btype_hlcm==3]
                    alts4 = alternatives[alternatives.btype_hlcm==4]
                    pdf1 = pd.DataFrame(index=alts1.index)
                    pdf2 = pd.DataFrame(index=alts2.index) 
                    pdf3 = pd.DataFrame(index=alts3.index)
                    pdf4 = pd.DataFrame(index=alts4.index)
                    
                    segments = movers.groupby(agents_groupby_txt)
                    
                    ind_vars1=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_residential','in_denver',]
                    ind_vars2=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_residential','in_denver',]
                    ind_vars3=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_residential','in_denver',]
                    ind_vars4=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_residential','in_denver',]
                    
                    for name, segment in segments:
                        if name == 1:
                            alts = alts1
                            ind_vars = ind_vars1
                        if name == 2:
                            alts = alts2
                            ind_vars = ind_vars2
                        if name == 3:
                            alts = alts3
                            ind_vars = ind_vars3
                        if name == 4:
                            alts = alts4
                            ind_vars = ind_vars4
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
                    for name, segment in segments:
                        name_coeff = str(name)
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
            
                ############     RDPLCM SIMULATION
                if core_components_to_run['Developer']:
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
                        logger.log_status('Employment diffs')
                        logger.log_status(emp_diff_county)
                        logger.log_status('Resunit diffs')
                        logger.log_status(ru_diff_county)
                        logger.log_status('NR sqft diffs')
                        logger.log_status(nr_diff_county)
                    
            elapsed = time.time() - seconds_start
            print "TOTAL elapsed time: " + str(elapsed) + " seconds."
            
        print 'Done running all three scenarios.'