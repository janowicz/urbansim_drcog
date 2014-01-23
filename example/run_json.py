import os, json, sys, time
from synthicity.utils import misc
import dataset

args = sys.argv[1:]

dset = dataset.DRCOGDataset(os.path.join(misc.data_dir(),'drcog.h5'))
num = misc.get_run_number()

if __name__ == '__main__':
  print time.ctime()
  for arg in args: 
      print 'STEP is:  ' + arg
      if arg == 'hlcm.json':
          print 'yoyoyo'
          print dset.households_for_estimation
      misc.run_model(arg,dset,estimate=1)
  print time.ctime()
  t1 = time.time()
  numyears = 10
  for i in range(numyears):
    t2 = time.time()
    for arg in args: 
        year=2010+i
        print year
        print 'STEP is:  ' + arg
        misc.run_model(arg,dset,show=1,estimate=0,simulate=1,year=year)
    print "Time to simulate year %d = %.3f" % (i+1,time.time()-t2)
    print len(dset.households.index)
    print dset.establishments.employees.sum()
    print dset.buildings.residential_units.sum()
    print dset.buildings.non_residential_sqft.sum()
  print "Actual time to simulate per year = %.3f" % ((time.time()-t1)/float(numyears))
  #from multiprocessing import Pool
  #pool = Pool(processes=len(args))
  #pool.map(run_model,args)
  dset.save_coeffs(os.path.join(misc.runs_dir(),'run_drive_%d.h5'%num))
  dset.save_output(os.path.join(misc.runs_dir(),'run_drive_%d.h5'%num))
  print time.ctime()
