import numpy as np, pandas as pd, os, statsmodels.api as sm
import synthicity.urbansim.interaction as interaction
from synthicity.utils import misc
import dataset, copy, time
dset = dataset.DRCOGDataset(os.path.join(misc.data_dir(),'drcog.h5'))
#VARIABLE LIBRARY
#parcel
p = dset.fetch('parcels')
#building
b = dset.fetch('buildings')
b['zone_id'] = p.zone_id[b.parcel_id].values
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
#merge parcels with zones
pz = pd.merge(p,z,left_on='zone_id',right_index=True)
pz['in_denver'] = (pz.county_id==8031).astype('int32')
#merge buildings with parcels/zones
dset.d['buildings'] = pd.merge(b,pz,left_on='parcel_id',right_index=True)

######
##ELCM
######

depvar = 'building_id'
SAMPLE_SIZE=50
choosers = dset.fetch('establishments')
choosers = choosers[(choosers.building_id>0)*(choosers.home_based_status==0)]
#choosers = choosers.ix[np.random.choice(choosers.index, 150000,replace=False)]
output_csv, output_title, coeff_name, output_varname = ("drcog-coeff-elcm-%s.csv","DRCOG EMPLOYMENT LOCATION CHOICE MODELS (%s)","emp_location_%s","establishment_building_ids")

#alts = dset.buildings[(dset.buildings.non_residential_sqft>0)]
alts = dset.buildings
ind_vars1=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
ind_vars2=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
ind_vars3=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
ind_vars4=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
ind_vars5=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
ind_vars6=['residential_units_zone','non_residential_sqft_zone','allpurpose_agglosum','acreage','unit_price_non_residential','in_denver',]
segments = choosers.groupby(['sector_id_six',])
for name, segment in segments:
    print name
    print len(segment.index)
    if name == 1:
        ind_vars = ind_vars1
        continue
    if name == 2:
        ind_vars = ind_vars2 
        continue
    if name == 3:
        ind_vars = ind_vars3 
        continue
    if name == 4:
        ind_vars = ind_vars4 
        continue
    if name == 5:
        ind_vars = ind_vars5 
        continue
    if name == 6:
        ind_vars = ind_vars6 #small estabs are price sensitive
    name = str(name)
    tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
    if len(segment.building_id) > 200: #reduce size of segment if too big so things don't bog down
        segment = segment.ix[np.random.choice(segment.index, 200,replace=False)]
    print segment.building_id
    print segment.building_id.describe()
    print alts.reset_index().building_id
    print alts.reset_index().building_id.describe()
    print SAMPLE_SIZE
    print segment[depvar]
    print segment.building_id
    print alts.reset_index().building_id.describe()
    print segment.reset_index().building_id.describe()
    alts.index = alts.index.astype('int32')
    #sample, alternative_sample, est_params = interaction.mnl_interaction_dataset(segment,alts,SAMPLE_SIZE,chosenalts=segment[depvar],weight_var='non_residential_sqft')
    sample, alternative_sample, est_params = interaction.mnl_interaction_dataset(segment,alts,SAMPLE_SIZE,chosenalts=segment[depvar])
    #alternative_sample['paris_x_employees'] = (alternative_sample.in_paris*alternative_sample.employees)
    print "Estimating parameters for segment = %s, size = %d" % (name, len(segment.index)) 
    if len(segment.index) > 50:
        est_data = pd.DataFrame(index=alternative_sample.index)
        for varname in ind_vars:
            est_data[varname] = alternative_sample[varname]
        est_data = est_data.fillna(0)
        data = est_data.as_matrix()
        try:
            fit, results = interaction.estimate(data, est_params, SAMPLE_SIZE)
            #print fit
            #print results
            fnames = interaction.add_fnames(ind_vars,est_params)
            print misc.resultstotable(fnames,results)
            misc.resultstocsv(fit,fnames,results,tmp_outcsv,tblname=tmp_outtitle)
            dset.store_coeff(tmp_coeffname,zip(*results)[0],fnames)
        except:
            print 'SINGULAR MATRIX OR OTHER DATA/ESTIMATION PROBLEM'
    else:
        print 'SAMPLE SIZE TOO SMALL'
print dset.coeffs[('emp_location_6','coeffs')]
print dset.coeffs[('emp_location_6','coeffs')][0]
print dset.coeffs[('emp_location_6','coeffs')][1]
print dset.coeffs[('emp_location_6','coeffs')][2]
print dset.coeffs[('emp_location_6','coeffs')][3]
print dset.coeffs[('emp_location_6','coeffs')][4]
print dset.coeffs[('emp_location_6','coeffs')][5]