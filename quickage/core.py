
##################
## MAIN CODE ###
##############

#imports
import os
import astropy.coordinates as astro_coord
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
#from astropy.constants import G

#gala imports
#import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
#from gala.units import galactic
import pandas as pd
from astropy.io import ascii
import matplotlib as mpl
from .plots import seaborn 
import math

#set paths
CODE_PATH = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
DATA_FOLDER= CODE_PATH + '/data/'

#define a coordinate frame
#not sure where this system came from ? Sarah's values 
#I have used this system to compute Jz so can't change it really
_ = astro_coord.galactocentric_frame_defaults.set('v4.0')

#v_sun = astro_coord.CartesianDifferential([11.1, 220 + 24.0, 7.25]*u.km/u.s)
#galcen_frame =astro_coord.Galactocentric(galcen_distance=8.2*u.kpc,
#                                    galcen_v_sun=v_sun)

#default coordinate frame
#sharma coordinate frame https://www.galah-survey.org/dr3/the_catalogues/#ages-masses-distances-and-other-parameters-estimated-by-bstep
v_sun = astro_coord.CartesianDifferential([11.1, 248., 7.25]*u.km/u.s) #almost the same as my coordinate 
galcen_frame =astro_coord.Galactocentric(galcen_distance=8.2*u.kpc,
                                    galcen_v_sun=v_sun)
#potential
pot=gp.MilkyWayPotential() #they use MCMillan et al potential but not sure if Adrian does
H = gp.Hamiltonian(pot)

def load_sharma_galah_sample():
    return pd.read_csv(DATA_FOLDER+'/galah_sharma.csv.gz')

def load_my_galah_sample():
    #note that this sample uses a different coordinate system
    return pd.read_csv(DATA_FOLDER+'/galah_lite.csv.gz')

def load_schneider_sample():
    #note that this sample uses a different coordinate system
    return pd.read_csv(DATA_FOLDER+'/schneiderdata_lite.csv.gz')

def get_phase_space(ra, dec, pmracosdec, pmdec, distance, rv ):
    """
	get phase space position of an object in our coordinate frame
    ra, dec in degree
    proper motions in mas/yr
    distance in pc
    rv in km/s
    """
    coord=astro_coord.SkyCoord(ra=ra*u.degree, dec=dec*u.degree,  
               pm_ra_cosdec= pmracosdec *u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr, \
               distance=distance*u.pc, 
              radial_velocity= rv*u.km/u.s)
    #phase space position
    pos=gd.PhaseSpacePosition(coord.transform_to(galcen_frame).cartesian)
    
    return coord, pos

def compute_actions(pos, plot_all_orbit=False, alpha=.1, print_pericenter=False):
    """
	Purpose compute actions based on the orbit of a star

	These default settings are used for all the samples, 
	they sample at least one orbit of the majority of the stars
    
    These are the default paramaters that I used to compute actions on the main samples

	"""
    nsteps=500
    time_dict= {'t':np.linspace(0*u.Myr, 1.0*u.Gyr, int(nsteps))}
    orbit=gp.Hamiltonian(pot).integrate_orbit(pos, **time_dict)
    #plot 
    orbit_to_plot=orbit[:,0]
    oplot=None
    if plot_all_orbit: 
        orbit_to_plot=orbit
        oplot=orbit_to_plot.cylindrical.plot( components=['rho', 'z', 'v_z'],  \
            units=[u.pc, u.pc, u.km/u.s] ,alpha=alpha, c='#0074D9')
    #documentation: http://gala.adrian.pw/en/latest/dynamics/actionangle.html
    toy_potential = gd.fit_isochrone(orbit[:,0])
    #print (toy_potential)
    #print (np.shape(orbit.z))
    #result = [gd.find_actions(orbit[:,idx], N_max=10, toy_potential=toy_potential) \
    #          for idx in tqdm(np.arange(np.shape(orbit)[-1]))]
    #new in gala 1.5 
    result= gd.find_actions(orbit, N_max=10, toy_potential=toy_potential)
    #print (result)
    if  print_pericenter:
        apos=[orbit[:,idx].apocenter() for idx in tqdm(np.arange(np.shape(orbit)[-1]))]
        peris=[orbit[:,idx].pericenter() for idx in tqdm(np.arange(np.shape(orbit)[-1]))]
        print ('apocenter --- {} +/- {}'.format(np.nanmedian(u.Quantity(apos)),\
                                                np.nanstd(u.Quantity(apos))))
        print ('pericenter --- {} +/- {}'.format(np.nanmedian(u.Quantity(peris)),\
                                                 np.nanstd(u.Quantity(peris))))
    return result, oplot

def estimate_age(source_coord, source_metal, nsigma=3, \
    select_by=['velocity', 'metallicity'], norbits=100,\
     plot_kde=False, data_set='galah', plot=False, volume=None, \
     velocity_volume=None, vertical_volume=None, file_plot=None, file_data=None, 
     export_data=False, weighted=False, \
     fweights=20, limits_weights=None, plot_orbits=False, cmap='viridis', vertical_velocity_volume=None):
    """
 	source_coord must be a dictionary with the following keywords
 	ra: ra in degree
 	dec: dec in degree
 	pmra: tuple in mas/yr (pmra, pmra_unc)
 	pmdec: tuple in mas/yr (pmdec, pmdec_unc)
 	distance: must a distance in pc (dist, dist_unc)
 	rv: tuple in km/s (rv, rv_unc)
    volume: radius physical volume in distance to select 
    vertical_volume: you can select by just abs(z)
    velocity_volume: radius velocity volume in distance to select 
 	source_metal: a tuple of [Fe/H] and uncertainty
    fweights: the number of volume steps to consider during weighting
    limits_weights: dictionary (dmin, dmax, vmin , vmax for age weighing 
    cmap: color map for plotting

 	use_jz: keyword to use vertical action as additional constraints


 	returns: age and posterior plots
 	uncertainties must be reasonable, not zero
 	"""
    
    #compute source coordinate 
    Scoord={'ra':source_coord['ra'], \
              'dec': source_coord['dec'],\
              'pmra':np.random.normal(source_coord['pmra'][0],source_coord['pmra'][1], int(norbits)),
              'pmdec':np.random.normal(source_coord['pmdec'][0],source_coord['pmdec'][1], int(norbits)),
              'distance':np.random.normal(source_coord['distance'][0],source_coord['distance'][1], int(norbits)),
              'rv': np.random.normal(source_coord['rv'][0],source_coord['rv'][1], int(norbits))}

    source_coord, source_pos=get_phase_space(Scoord['ra'], 	Scoord['dec'],\
                       	Scoord['pmra']*np.cos(	Scoord['dec']*u.degree), \
                       	Scoord['pmdec'], 	Scoord['distance'], Scoord['rv'])

    use_jz = 'actions' in select_by
    #get data
    if isinstance(data_set, str):
        if data_set=='galah': 
            data= load_sharma_galah_sample()

        if data_set=='schneider': 
            data=load_schneider_sample()

    #option to pass custom dataset to bypass reading files
    if not isinstance(data_set, str):
        data=data_set
        
    #if use jz
    total_cut=[]

    if use_jz:
        source_res=compute_actions(source_pos, plot_all_orbit=plot_orbits)
        #source_actions=np.vstack(source_res[0]['actions'].apply(lambda x: np.array(x)).values)
        source_actions= source_res[0]['actions']
        mean_source_jr= np.nanmedian(source_actions[0])
        mean_source_lz= np.nanmedian(source_actions[1])
        mean_source_jz= np.nanmedian(source_actions[-1]) #conversion from (u.kpc**2/u.Gyr).to(u.km*u.kpc/(u.s))
        std_source_jz= np.nanstd(source_actions[-1])
   		#forget about angles and frequencies
        print ('radial action (Jr) {:.5e} +/- {:.5e} kpc km/s'.format(np.nanmedian(source_actions[0]), np.nanstd(source_actions[0])))
        print ('vertical angular momentum (Lz) {:.5e} +/- {:.5e} kpc km/s'.format(np.nanmedian(source_actions[1]), np.nanstd(source_actions[1])))
        print ('vertical action (Jz) {:.5e} +/- {:.5e} kpc km/s'.format(mean_source_jz, std_source_jz))

   		#compute boolean vertical_actions within uncertainties
        #jz_cut= np.logical_and(data.Jz < mean_source_jz+ nsigma* std_source_jz, \
        #	data.Jz > mean_source_jz- nsigma* std_source_jz)
        jz_cut=data.Jz < (mean_source_jz+ nsigma* std_source_jz)
        total_cut.append(jz_cut)

   	#kinematics, metallicity cuts
   	#total velocity of the source
    source_total_v=(source_coord.transform_to(galcen_frame).v_x**2+
            source_coord.transform_to(galcen_frame).v_y**2+
                source_coord.transform_to(galcen_frame).v_z**2)**0.5
    
    source_x=np.nanmedian(source_coord.transform_to(galcen_frame).x.to(u.pc)).value

    source_y=np.nanmedian(source_coord.transform_to(galcen_frame).y.to(u.pc)).value

    source_z= np.nanmedian(source_coord.transform_to(galcen_frame).z.to(u.pc)).value

    vr= ((source_coord.transform_to(galcen_frame).v_x**2+
                source_coord.transform_to(galcen_frame).v_y**2)**0.5).value
                 
    vz=(source_coord.transform_to(galcen_frame).v_z).value

    vx=source_coord.transform_to(galcen_frame).v_x.value
    vy=source_coord.transform_to(galcen_frame).v_y.value


   	#compute only kinematics within the velocity ellipse around the star
    kinematic_cut=data.vtot < (np.nanmedian(source_total_v).value)


    vz_cut=data.v_z < np.nanmedian(vz)

	#metallicity within n-sigma
    metallicity_cut= np.logical_and(data['fe_h'].values > source_metal[0]-nsigma*source_metal[-1],
									data['fe_h'].values < source_metal[0]+nsigma*source_metal[-1])

    if 'metallicity' in select_by:
        total_cut.append(metallicity_cut)

    if 'velocity' in select_by:
        total_cut.append(kinematic_cut)

    if 'vertical_velocity' in select_by:
        total_cut.append(vz_cut)

    #select by volume (only look at stars within x pc of our target)
    if volume is not None:
        volume_cut= (source_x-data.x)**2+  (source_y-data.y)**2+ (source_z-data.z)**2 < volume**2
        total_cut.append(volume_cut)

    if velocity_volume is not None:
        velocity_volume_cut=(np.nanmedian(vr)-data.v_r)**2+ (np.nanmedian(vz)-data.v_z)**2 < velocity_volume**2
        total_cut.append(velocity_volume_cut)

    if vertical_velocity_volume is not None:
        vert_volume_cut= (np.nanmedian(vz)-data.v_z)**2 < vertical_velocity_volume**2
        total_cut.append(vert_volume_cut)

    if vertical_volume is not None:
        vert_volume_cut= (source_z-data.z)**2 < vertical_volume**2
        total_cut.append(vert_volume_cut)


	#selection
    nans= np.isnan(data.age_bstep)
    total_cut.append(~nans)
    selection= np.logical_and.reduce(total_cut)
    MEDIAN_AGE=np.nanmedian(data.age_bstep.values[selection])
    STD_AGE=[np.percentile(data.age_bstep.values[selection], 16), \
	         np.percentile(data.age_bstep.values[selection], 84)]

    age_samples=data.age_bstep[selection].values
    age_weights=np.ones_like(age_samples)

    if weighted:
        center={'x':source_x, 'y':source_y, 'z':source_z,  'r': (source_x**2+ source_y**2)**0.5,\
        'v_x': np.nanmedian(vx), 'v_y':np.nanmedian(vy), 'v_z':np.nanmedian(vz), \
        'v_r': (np.nanmedian(vy)**2+np.nanmedian(vx)**2)**0.5 }
        MEDIAN_AGE,  STD_AGE, age_samples,age_weights =get_volume_weighted_age(center, data, \
            nsteps=fweights, limits= limits_weights)

    #print (data.columns)

    if plot:
        fig, ax=plt.subplots( figsize=(8, 6))
        if not  plot_kde:
            _=ax.hist(data.age_bstep, histtype='step', bins='auto', lw=3, density=True, \
                      linestyle='--', color='#AAAAAA', label='Full sample')
            _=ax.hist(age_samples,  lw=3, density=True, \
                      linestyle='-', histtype='step', color='#111111', bins=32, label='Selected', weights=age_weights)
        if plot_kde:
            _= seaborn.kdeplot(data.age_bstep.values, lw=3, linestyle='--', color='#AAAAAA', label='Full sample',\
             ax=ax, common_grid=True, multiple="stack", alpha=0.5, cut=0)
            _= seaborn.kdeplot(age_samples, lw=4, linestyle='-', color='#111111', \
                label='Selected', ax=ax, common_grid=True,multiple="stack", alpha=0.5, cut=0, \
                weights=age_weights)


        ax.axvspan(STD_AGE[0], STD_AGE[-1], alpha=0.2, color='#01FF70')
        ax.axvline(MEDIAN_AGE, color='#FFDC00')
        ax.set(xlabel='Age (Gyr)', ylabel='Normalized Density')
        plt.legend()
        ax.minorticks_on()
        plt.savefig(file_plot)

        #print(data.columns)
        
        fig, ax=plt.subplots(ncols=2, figsize=(12, 4))
        #ax[0].scatter(data.v_r, data.v_z, s=0.5,   c='k', \
        #              marker='+',  cmap='winter', vmin=0, vmax=13)
        ax[0].scatter(data.v_r, data.v_z, s=1.,  alpha=0.1,  c=data.age_bstep, \
                      marker='+',  cmap=cmap, vmin=0, vmax=13)

        ax[0].errorbar(np.nanmedian(vr), np.nanmedian(vz), xerr=np.nanstd(vr), \
            yerr=np.nanstd(vz), fmt='o', ms=15, c='k')

        ax[0].set(xlabel=r'$( V_x^2 + V_y ^2)^{0.5}$ (km/s) ', ylabel=r'$V_z$ (km/s)', xlim=[0, 500], ylim=[-500, 500])

        #plot Jz instead
        if use_jz:    
            ax[1].scatter(data['fe_h'],  data.Jz, s=0.5,  c=data.age_bstep, \
                          marker='+',  cmap=cmap, vmin=0, vmax=13)
            ax[1].errorbar(source_metal[0], mean_source_jz, xerr=source_metal[-1],\
                       yerr=std_source_jz, marker='o', ms=15, c='k')
            ax[1].set(  xlabel='[Fe/H]', \
               ylabel=r'J$_z$ (kpc km/s) ', ylim=[-0.1, 1.1])
        

        #plot vertical velocity instead   
        if not use_jz:
            #ax[1].scatter(data['fe_h'],  data['v_z'], s=0.5,  c='k', \
            #              marker='+')
            ax[1].scatter(data['fe_h'],  data['v_z'], s=0.5, alpha=0.1, c=data.age_bstep, \
                          marker='+',  cmap=cmap, vmin=0, vmax=13)
            ax[1].errorbar(source_metal[0], np.nanmedian(vz), xerr=source_metal[-1],\
                       yerr=np.nanstd(vz), marker='o', ms=15, c='k')
            ax[1].set(  xlabel='[Fe/H]', \
               ylabel=r'V$_z$ (km/s) ', ylim=[-500, 500])
        
        
        norm= mpl.colors.Normalize(vmin=0,vmax=13)
        mp=mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cax = fig.add_axes([1.01, 0.25, .015, 0.7])
        cbar=plt.colorbar(mp, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel(r'Age (Gyr)', fontsize=18)
        plt.tight_layout()
        for a in ax: a.minorticks_on()
        plt.savefig(file_plot.replace('.', '_scatter_'), rasterized=True, bbox_inches='tight')

    print ('Age {:.2f} - {:.2f} + {:.2f} Gyr'.format(MEDIAN_AGE,  MEDIAN_AGE-STD_AGE[0], STD_AGE[1]-MEDIAN_AGE))
    return { 'median_age':MEDIAN_AGE,
			'std_age': (MEDIAN_AGE-STD_AGE[0], STD_AGE[1]-MEDIAN_AGE),
			'posterior': age_samples,
            'weights': age_weights,
            'coords': source_coord.transform_to(galcen_frame)  }

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    #remove nans
    nans= np.logical_or(np.isnan(values), np.isnan(weights))
    values=values[~nans]
    weights=weights[~nans]
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def draw_around(center, condition, data):
    #condition is a dictionary
    booleans=[]
    for k in condition.keys():
        #exclude the star itself (if it happens to be in the sample) (things within .1 percent)
        exclude= np.abs(data[k]-center[k]) <= 0.001*np.abs(center[k])
        booleans.append(~exclude)
        booleans.append(np.abs(data[k]-center[k])<condition[k])
        
    comb_bool=np.logical_and.reduce(booleans)
    med= np.nanmedian(data.age_bstep[comb_bool])
    unc= np.nanstd(data.age_bstep[comb_bool])
    return med, unc

def compute_age_around(center, nsteps, data, dmin=10, dmax=1000, vmin=10, vmax=100):
    """
    center is the star that we want to compute ages around
    additional_cuts are boolean
    data is the data
    """
    weights=[]
    samples=[]
    for d in tqdm(np.logspace(np.log10(dmin), np.log10(dmax), nsteps)): #max 1 kpc, min 20 pc
        for v in (np.logspace(np.log10(vmin), np.log10(vmax), nsteps)): #max 500 km/s, min 20 km/s
            w=1/(d**3*v**3)
            conds={'r':d,
               'z':d,
               'v_r':v,
               'v_z':v}
            #if additional_cuts != None:
            #    for k in additional_cuts.keys(): conds[k]=additional_cuts[k]
            med, std=draw_around(center, conds, data)
            samples.append(np.random.normal(med, std, 1))
            weights.append(w)
    #normalize weights
    weights=np.array(weights)/np.nanmax(weights)
    return {'samples': np.array(samples), 'weights': np.array(weights)}

def get_volume_weighted_age(center, data, nsteps=20, limits=None):
    #set some defaults
    data['r']= (data['x']**2+ data['y']**2)**0.5
    if limits==None:
        limits= {'dmin': 10, 'dmax': 1000, 'vmin': 10, 'vmax': 400}
    res=compute_age_around(center, nsteps, data, **limits)
    value, unc=weighted_avg_and_std(res['samples'].flatten(), res['weights'])
    return value, [value-unc, value+unc], np.array(res['samples'].flatten()), np.array(res['weights'])



def load_schneider_samples_old(use_jz=False):
    """
    Read bensby et al, luck et al, cassagrande et al., spocs
    
    """
    bensby=ascii.read(DATA_FOLDER+'/bensby_30pc_gaia1.txt', \
                            names ='ra              de    \
                                    [Fe/H]       age1     l_age1     u_age1       \
                                    ra_gaia              de_gaia        plx       eplx    \
                                    pmra      epmra       pmde      epmde      \
                                    rv        erv'.split()).to_pandas()
    luck=ascii.read(DATA_FOLDER+'/luck_30pc_gaia1.txt', \
                   names='ra              de     [Fe/H]       age1     l_age1    \
                   u_age1         ra_gaia         de_gaia        plx       eplx      \
                   pmra      epmra       pmde      epmde         rv   \
                   erv '.split()).to_pandas()
                   
    casgr=ascii.read(DATA_FOLDER+'/casagrande_30pc_gaia1.txt', \
                     names='ra              de     [Fe/H]      ageMP     age16P     \
                     age84P              ra_gaia              de_gaia        \
                     plx       eplx       pmra      epmra       pmde      \
                     epmde         rv        erv'.split()).to_pandas()
    
    spocs=ascii.read(DATA_FOLDER+'/spocs_30pc_gaia1.txt', \
                     names=' ra              de     [Fe/H]       age1     l_age1     u_age1          \
                     ra_gaia              de_gaia        plx       eplx       pmra   \
                     epmra       pmde      epmde         rv        erv '.split()).to_pandas()
            
    luck['pmracosdec']= luck.pmra*np.cos(luck.de*u.degree)
    bensby['pmracosdec']= bensby.pmra*np.cos(bensby.de*u.degree)
    spocs['pmracosdec']= spocs.pmra*np.cos(spocs.de*u.degree)
    casgr['pmracosdec']= casgr.pmra*np.cos(casgr.de*u.degree)
    
    data=pd.concat([bensby, luck, spocs, casgr]).rename(columns={'age1':'age_bstep'})
    
    data_coord, data_pos=get_phase_space(data.ra_gaia.values, data.de_gaia.values, data.pmracosdec.values, data.pmde.values, 1000/data.plx.values, data.rv.values )
    
    data['vtot']=((data_coord.transform_to(galcen_frame).v_x**2+data_coord.transform_to(galcen_frame).v_y**2+data_coord.transform_to(galcen_frame).v_z**2)**0.5).value


    data['v_x']=data_coord.transform_to(galcen_frame).v_x.value
    data['v_y']=data_coord.transform_to(galcen_frame).v_y.value
    data['v_z']=data_coord.transform_to(galcen_frame).v_z.value
    data['v_r']=(data.v_x**2+data.v_y**2)**0.5
    data['x']=data_coord.transform_to(galcen_frame).x.to(u.pc).value
    data['y']=data_coord.transform_to(galcen_frame).y.to(u.pc).value
    data['z']=data_coord.transform_to(galcen_frame).z.to(u.pc).value
    data['r']= (data.x**2 + data.y**2)**0.5

    #compute actions if necessary
    if use_jz:
        data_res=compute_actions(data_pos, plot_all_orbit=False)
        #data_actions=np.vstack(data_res[0]['actions'].apply(lambda x: np.array(x)).values)
        data_actions= data_res[0]['actions']
        #data_angles=np.vstack(data_res[0]['angles'].apply(lambda x: np.array(x)).values)
        #data_freqs=np.vstack(data_res[0]['freqs'].apply(lambda x: np.array(x)).values)
        data['Jr']=data_actions[:,0]#*1000 #units (kpc$^2$/Myr)
        data['Jphi']=data_actions[:,1]#*1000 
        data['Jz']=data_actions[:,2]#*1000
    return data.rename(columns={'[Fe/H]': 'fe_h', 'de': 'dec', 'pmde': 'pmdec'})
