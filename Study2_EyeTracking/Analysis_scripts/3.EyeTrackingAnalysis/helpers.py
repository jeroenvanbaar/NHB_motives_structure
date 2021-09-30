import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy, os, glob, sys, re
from matplotlib import image
sys.path.append('pygazeanalyser')
from gazeplotter import parse_fixations, gaussian

def load_data(sub, session = None, part = None, in_concise = False, out_concise = True,
              base_dir = '/Users/jvanbaar/Dropbox (Brown)/Postdoc FHL/JEROEN/SOC_STRUCT_LEARN'):
    
    # In_concise: use if only stored select columns in IDF converter
    # out_concise: use if you only want pupil diam (mm) and gaze pos (point of regard POS) (px) (recommended)
    # If you leave session or part empty, will load all data
    
    cols_concise = ['Time','L Raw X [px]','L Mapped Diameter [mm]','R Mapped Diameter [mm]',
               'L POR X [px]','L POR Y [px]','R POR X [px]','R POR Y [px]']
    
    if (session is None) | (part is None):
        filename_pattern = (base_dir+
                '/Study2_EyeTracking/Data/Cleaned/Gaze_full_info/sub-%03d_ses-*_part* Samples.txt'%sub)
        filenames = glob.glob(filename_pattern)
        filenames.sort()
        print('Loading file%s:'%('s' if len(filenames) > 1 else ''))
        [print(filenames[i]) for i in range(len(filenames))]
        sessions = [int(re.findall('ses-\d\d\d',filenames[i])[0][-3:]) 
                    for i in range(len(filenames))]
        parts = [int(re.findall('part\d',filenames[i])[0][-1:]) 
                    for i in range(len(filenames))]
        
    else:
        filename = (base_dir +
        '/Study2_EyeTracking/Data/Cleaned/sub_%03d/sub-%03d_ses-%03d_part%i Samples.txt'%(
            sub, sub, session, part))
        print('Loading file %s'%filename)
        filenames = [filename]
        sessions = [session]
        parts = [part]
    
    gaze_data_all = pd.DataFrame()
    for fi,filename in enumerate(filenames):
        if out_concise:
            gaze_data = pd.read_csv(filename, sep = '\t', header = 0, skiprows = 38, usecols = cols_concise)
        else:
            gaze_data = pd.read_csv(filename, sep = '\t', header = 0, skiprows = 38)
        print('Raw data shape is ',end='')
        print(gaze_data.shape)
        
        if in_concise:
            msg_col = 'L Mapped Diameter [mm]'
        else:
            rename_cols = ['time','L Raw X [px]','L_diam','R_diam','L_X','L_Y','R_X','R_Y']
            gaze_data.columns = rename_cols
            msg_col = 'L Raw X [px]'
            
        gaze_data['dt'] = (gaze_data['time'] - gaze_data.loc[0,'time'])/1000
        gaze_data['sub'] = sub

        gaze_data['chunk'] = gaze_data[msg_col].apply(lambda x: x[11:] if 'Message' in x else np.nan)
        gaze_data['chunk'].interpolate(method='pad',inplace=True)
        gaze_data = gaze_data.dropna().reset_index(drop=True) # This drops all message lines
        gaze_data.drop(columns=msg_col, inplace=True)
        gaze_data.iloc[:,1:-2] = gaze_data.iloc[:,1:-2].astype(float) # Sets all cols to float except time, sub and chunk

        gaze_data['session'] = sessions[fi]
        gaze_data['part'] = parts[fi]

        gaze_data_all = gaze_data_all.append(gaze_data).reset_index(drop=True)
        
    print('Processed data shape is ',end='')
    print(gaze_data_all.shape)

    return gaze_data_all

def parse_calibrations(gaze_data):
    gaze_data_calib = gaze_data.loc[
        (gaze_data['chunk'].str.find('Now get to location') > -1),:].copy()
    
    gaze_data_calib['target_X'] = gaze_data_calib['chunk'].apply(
        lambda x: int(x[x.find('x = ')+4:x.find('x = ')+8].strip(',')))
    gaze_data_calib['target_Y'] = gaze_data_calib['chunk'].apply(
        lambda x: int(x[x.find('y = ')+4:x.find('y = ')+8].strip(',')))
    
    return gaze_data_calib

def plot_calibration(gaze_data_calib, store_to_disk = True, folder = '/Users/jvanbaar/Desktop',
             dot_alpha = .1, dot_size = 75, return_figs = False, show_timecourse = False):

    sub = gaze_data_calib['sub'].unique()
    if len(sub) > 1:
        ValueError('More than one subject in gaze_data_calib')
    
    if return_figs:
        figs = []

    if store_to_disk:
        filepath = folder + '/gaze_sub-%03d'%sub
        if not os.path.exists(filepath):
            os.mkdir(filepath)
            print('made dir %s'%filepath)

    colors = sns.color_palette('tab10',8)
        
    for part in gaze_data_calib['part'].unique():
        
        gaze_data_calib_part = gaze_data_calib.query('part == @part').copy()
        
        print('Showing calibration data for part %i'%part)

        target_locations = gaze_data_calib_part[['target_X','target_Y']].drop_duplicates(
            ).copy().reset_index(drop=True)

        fig, ax = plt.subplots(1,1,figsize=[8,8])
        ax.set(xlim = [0,1680], ylim = [0,1050], aspect = 1)

        for ti,target in target_locations.iterrows():
            X = target['target_X']
            Y = target['target_Y']
#             Y = target['target_Y']

            plot_dat = gaze_data_calib_part.query('target_X == @X & target_Y == @Y')[
                ['dt','L_X','L_Y','R_X','R_Y']]
            plot_dat['L_Y'] = 1050 - plot_dat['L_Y']
            plot_dat['R_Y'] = 1050 - plot_dat['R_Y']
            plot_dat['ddt'] = plot_dat['dt'] - plot_dat.iloc[0]['dt']
            plot_dat = plot_dat.query('ddt > 300')
            ax.scatter(np.mean(plot_dat[['R_X','L_X']],axis=1),
                np.mean(plot_dat[['R_Y','L_Y']],axis=1),
                c = [colors[ti]],
                alpha = dot_alpha, s = dot_size, linewidth = 0, edgecolor = 'k')
            ax.scatter(X, 1050 - Y, color = 'r', s = 75, marker='o',
                       linewidth = 2, edgecolor = 'k')
    #         ax.text(X, Y, '%i'%ti, fontdict = {'fontsize':10,
    #             'verticalalignment':'center','horizontalalignment':'center'})

        ax.set(title = 'Subject %03d - calibration part %i'%(sub,part),
               xlabel = 'X', ylabel = 'Y')
        if store_to_disk:
            plt.savefig(filepath + '/calibration_part-%i.pdf'%part,
                bbox_inches='tight', transparent = True)

        if return_figs:
            figs.append(fig)

        plt.show()

        if show_timecourse:

            fig, axes = plt.subplots(2,4,figsize=[12,4])
            for ai, ax in enumerate(axes.ravel()):
                target = target_locations.iloc[ai]
                X = target['target_X']
                Y = target['target_Y']
                plotDat = gaze_data_calib_part.query('target_X == @X & target_Y == @Y')[
                    ['dt','L_X','L_Y','R_X','R_Y']]
                plot_dat['L_Y'] = 1050 - plot_dat['L_Y']
                plot_dat['R_Y'] = 1050 - plot_dat['R_Y']
                plotDat['ddt'] = plotDat['dt'] - plotDat.iloc[0]['dt']
                plotDat.plot(x = 'ddt', y = ['L_X','L_Y','R_X','R_Y'], ax = ax)
                ax.set(xlabel = 'Time (ms)');
            plt.tight_layout()
            if store_to_disk:
                plt.savefig(filepath + '/calibration_timecourses_part-%i.pdf'%part,
                bbox_inches='tight', transparent = True)
            if return_figs:
                figs.append(fig)
            plt.show()

    if return_figs:
        return figs
    
def parse_trials(gaze_data, verbose = False):
    gaze_data['trial_row'] = gaze_data['chunk'].apply(lambda x: 'Starting block' in x)
    gaze_data_trials = gaze_data.query('trial_row').copy()
    
    gaze_data_trials['block'] = gaze_data_trials['chunk'].apply(
        lambda x: int(x.split(' ')[2].strip(',')))
    gaze_data_trials['trial'] = gaze_data_trials['chunk'].apply(
        lambda x: int(x.split(' ')[4].strip(',')))
    gaze_data_trials['S'] = gaze_data_trials['chunk'].apply(
        lambda x: int(x.split(' ')[7].strip(',')))
    gaze_data_trials['T'] = gaze_data_trials['chunk'].apply(
        lambda x: int(x.split(' ')[11].strip(',')))

    if verbose:
        print(gaze_data_trials[['sub','session','part','chunk','block','trial','S','T']
                              ].drop_duplicates().head())
        print(gaze_data_trials[['sub','session','part','chunk','block','trial','S','T']
                              ].drop_duplicates().tail())
        gaze_data_trials['session'].plot(figsize=[10,4])
        gaze_data_trials['part'].plot()
        gaze_data_trials['block'].plot()
        plt.legend(loc=[1.1,.5])
        plt.xlabel('Data line')
        plt.show()
        gaze_data_trials['trial'].plot(figsize=[10,4])
        gaze_data_trials['S'].plot()
        gaze_data_trials['T'].plot()
        plt.legend(loc=[1.1,.5])
        plt.xlabel('Data line')
        
    return gaze_data_trials

def get_number_locations():
    number_locations = pd.DataFrame([
        # Opponent:
        ['10', 507, 280], ['T', 1050, 280],
        ['S', 507, 695], ['5', 1050, 695],
        # Player:
        ['10', 250, 485], ['S', 813, 485],
        ['T', 250, 900],['5', 813, 900]])
    number_locations.columns = ['num','X','Y']
    # Since psychtoolbox draws from top to bottom, we need to invert the Y axis:
    number_locations['Y'] = number_locations['Y']

    return number_locations

def get_number_locations_indices():
    number_locations = pd.DataFrame([
        ['10_1', 250, 485], ['10_2', 507, 280],
        ['S_1', 813, 485], ['T_1', 1050, 280],
        ['T_2', 250, 900], ['S_2', 507, 695],
        ['5_1', 813, 900],['5_2', 1050, 695],])
    number_locations.columns = ['num','X','Y']
    # Since psychtoolbox draws from top to bottom, we need to invert the Y axis:
    number_locations['Y'] = number_locations['Y']

    return number_locations

# def my_draw_display(dispsize = [1680, 1050], dpi = 210, method = 'draw', imagefile = None, ax = None,
#                    S = None, T = None):

#     make_fig = True if ax is None else False
        
#     if method == 'draw':
        
#         if make_fig:
#             fig, ax = plt.subplots(1,1, figsize = [dispsize[0]/dpi, dispsize[1]/dpi])
            
#         ax.set(xlim = [0,dispsize[0]], ylim = [0,dispsize[1]], aspect = 1)
        
#         number_locations = get_number_locations()
#         if S is not None:
#             number_locations.loc[number_locations['num']=='S','num'] = S
#         if T is not None:
#             number_locations.loc[number_locations['num']=='T','num'] = T
#         for ri, row in number_locations.iterrows():
#             ax.text(row['X'],dispsize[1] - row['Y'],row['num'], fontdict = {'fontsize':20/(dpi/200),
#                 'verticalalignment':'center','horizontalalignment':'center',
#                 'color':'r'})
        
#         ax.set(xticks = [], yticks = [])
    
#     elif method == 'load':
        
#         if imagefile is None:
#             imagefile = ('/Users/jeroen/Dropbox (Brown)/Postdoc FHL/JEROEN/SOC_STRUCT_LEARN/Study2_EyeTracking/'+
#                 'Analysis_scripts/EyeTrackingAnalysis/game_screen.png')

#         # SET UP IMAGE ARRAY
#         _, ext = os.path.splitext(imagefile)
#         ext = ext.lower()
#         data_type = 'float32' if ext == '.png' else 'uint8'
#         screen = np.zeros((dispsize[1],dispsize[0],3), dtype=data_type)

#         # LOAD IMAGE AND PUT IN ARRAY
#         imagefile = base_dir + '/Study2_EyeTracking/Analysis_scripts/EyeTrackingAnalysis/game_screen.png'
#         img = image.imread(imagefile, format = 'png')[:,:,:3] # Leave out the alpha layer
#         h, w = img.shape[:2]
#         screen[:h,:w,:] += img
    
#         # CREATE FIGURE AND DRAW SCREEN
#         if make_fig:
#             figsize = (dispsize[0]/dpi, dispsize[1]/dpi)
#             fig, ax = plt.subplots(1,1, figsize=figsize, frameon=False)
#         ax.set_axis_off()
#         ax.axis([0,dispsize[0],0,dispsize[1]])
#         ax.imshow(screen)
    
#     return ax

# def my_draw_fixations(fix, dispsize, imagefile=None, dpi = 100, display_method = 'draw',
#                    durationsize=True, durationcolour=True, alpha=0.5, savefilename=None,
#                      S = None, T = None, ax = None):
    
#     ax = my_draw_display(dispsize = dispsize, dpi = dpi, method = display_method, imagefile = imagefile,
#                          ax = ax, S = S, T = T)
    
#     if durationsize:
#         siz = 1 * (fix['dur']/30.0)
#     else:
#         siz = 1 * numpy.median(fix['dur']/30.0)

#     if durationcolour:
#         col = fix['dur']
#     else:
#         col = sns.color_palette('tab10')[0]

#     # draw circles
#     ax.scatter(fix['x'],fix['y'], s=siz, c=col, marker='o', cmap='viridis', alpha=alpha, edgecolors='none')

#     # FINISH PLOT
#     # invert the y axis, as (0,0) is top left on a display
#     ax.invert_yaxis()
# #     # save the figure if a file name was provided
# #     if savefilename != None:
# #         fig.savefig(savefilename)
        
# def my_draw_heatmap(fix, dispsize, imagefile=None, dpi = 100, display_method = 'draw', ax = None,
#                  durationweight=True, alpha=0.5, savefilename=None, S = None, T = None):


#     # IMAGE
#     ax = my_draw_display(dispsize = dispsize, dpi = dpi, method = display_method, imagefile = imagefile,
#                          ax = ax, S = S, T = T)

#     # HEATMAP
#     # Gaussian
#     gwh = 200
#     gsdwh = gwh/6
#     gaus = gaussian(gwh,gsdwh)
#     # matrix of zeroes
#     strt = gwh/2
#     heatmapsize = int(dispsize[1] + 2*strt), int(dispsize[0] + 2*strt)
#     heatmap = np.zeros(heatmapsize, dtype=float)
#     strt = int(strt)
#     # create heatmap
#     for i in range(0,len(fix['dur'])):
#         # get x and y coordinates
#         #x and y - indexes of heatmap array. must be integers
#         x = strt + int(fix['x'][i]) - int(gwh/2)
#         y = strt + int(fix['y'][i]) - int(gwh/2)
#         # correct Gaussian size if either coordinate falls outside of
#         # display boundaries
#         if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
#             hadj=[0,gwh];vadj=[0,gwh]
#             if 0 > x:
#                 hadj[0] = abs(x)
#                 x = 0
#             elif dispsize[0] < x:
#                 hadj[1] = gwh - int(x-dispsize[0])
#             if 0 > y:
#                 vadj[0] = abs(y)
#                 y = 0
#             elif dispsize[1] < y:
#                 vadj[1] = gwh - int(y-dispsize[1])
#             # add adjusted Gaussian to the current heatmap
#             try:
#                 heatmap[y:y+vadj[1],x:x+hadj[1]] += gaus[vadj[0]:vadj[1],hadj[0]:hadj[1]] * fix['dur'][i]
#             except:
#                 # fixation was probably outside of display
#                 pass
#         else:
#             # add Gaussian to the current heatmap
#             heatmap[y:y+gwh,x:x+gwh] += gaus * fix['dur'][i]
#     # resize heatmap
#     heatmap = heatmap[strt:dispsize[1]+strt,strt:dispsize[0]+strt]
#     # remove zeros
#     lowbound = np.mean(heatmap[heatmap>0])
#     heatmap[heatmap<lowbound] = np.NaN
#     # draw heatmap on top of image
#     ax.imshow(heatmap, cmap='jet', alpha=alpha)

#     # FINISH PLOT
#     # invert the y axis, as (0,0) is top left on a display
#     ax.invert_yaxis()
#     # save the figure if a file name was provided
# #     if savefilename != None:
# #         fig.savefig(savefilename)

# #     return fig

def load_fixations(sub, session = None, part = None,
                   base_dir = '%s/Dropbox (Brown)/Postdoc FHL/JEROEN/SOC_STRUCT_LEARN'%os.environ['HOME']):
    
    if (session is None) | (part is None):
        filename_pattern = (base_dir+
                '/Study2_EyeTracking/Data/Cleaned/Gaze_events/sub-%03d_ses-*_part* Events.txt'%sub)
        filenames = glob.glob(filename_pattern)
        filenames.sort()
        print('Loading file%s:'%('s' if len(filenames) > 1 else ''))
        [print(filenames[i]) for i in range(len(filenames))]
        sessions = [int(re.findall('ses-\d\d\d',filenames[i])[0][-3:]) 
                    for i in range(len(filenames))]
        parts = [int(re.findall('part\d',filenames[i])[0][-1:]) 
                    for i in range(len(filenames))]
        print(sessions, parts)

        # gaze_events_all = pd.DataFrame()
        fixation_events_all = pd.DataFrame()
        for fi,filename in enumerate(filenames):
            gaze_events = pd.read_csv(filename, skiprows = 18, sep = '\n').iloc[:,0].str.split('\t', expand=True)
            print('Raw data shape is ',end='')
            print(gaze_events.shape)

            gaze_events = gaze_events.rename(columns={0:'event_type'})
            fixation_events = gaze_events.query('event_type == "Fixation L"').append(
                gaze_events.query('event_type == "Fixation R"')).iloc[:,:-4]
            fixation_events.columns = ['event_type','tracker_trial','number',
                                   'start','end','duration','X','Y','disp_X','disp_Y',
                                   'plane','avg_pupil_X','avg_pupil_Y']
            fixation_events[['start','end','duration']] = fixation_events[['start','end','duration']].astype(int)
            fixation_events[['X','Y','disp_X','disp_Y','avg_pupil_X','avg_pupil_Y']] = (
                fixation_events[['X','Y','disp_X','disp_Y','avg_pupil_X','avg_pupil_Y']].astype(float))
            fixation_events['file_index'] = fi
            fixation_events_all = fixation_events_all.append(fixation_events).reset_index(drop=True)
    
    return fixation_events_all

def select_trial_fixations(trial, gaze_data_trials, fixation_events, eye = 'L'):
    
    # Load trial data => get times
    trial_dat = gaze_data_trials.query('trial == @trial')
    
    # Fixations within these times
    trial_fixations = fixation_events.query('start >= %i & end <= %i'%(
        trial_dat['time'].min(), trial_dat['time'].max())).copy()
    
    # one eye:
    fix_select = trial_fixations.query('event_type == "Fixation %s"'%eye).copy(
        )[['start','end','duration','X','Y']].reset_index(drop=True)
    
    return fix_select

def get_heatmap(fix, dispsize):
    # HEATMAP
    # Gaussian
    gwh = 200
    gsdwh = gwh/6
    gaus = gaussian(gwh,gsdwh)
    # matrix of zeroes
    strt = int(gwh/2) # This provides padding on all sides of the heatmap, for fixations at the border?? "staart"??
    heatmapsize = int(dispsize[1] + 2*strt), int(dispsize[0] + 2*strt)
    heatmap = np.zeros(heatmapsize, dtype=float)

    # create heatmap
    for i in range(len(fix['dur'])):
        # get x and y coordinates
        #x and y - indexes of heatmap array. must be integers
        x = int(round(fix['x'][i]))
        y = int(round(fix['y'][i]))
        # only include fixation if it falls within display boundaries
        if (0 <= x <= dispsize[0]) and (0 <= y <= dispsize[1]):
            # add Gaussian to the current heatmap
            heatmap[y:y+gwh,x:x+gwh] += gaus * fix['dur'][i] / 1000000
    # crop heatmap to display
    heatmap = heatmap[strt:dispsize[1]+strt,strt:dispsize[0]+strt]
    
    return heatmap

def get_heatmap_fast(x, y, dur, dispsize):
    
    ### x, y, dur are 1-d numpy arrays of the same length

    # HEATMAP
    # Gaussian
    gwh = 200
    gsdwh = gwh/6
    gaus = gaussian(gwh,gsdwh)
    
    # matrix of zeroes
    strt = int(gwh/2) # This provides padding on all sides of the heatmap, for fixations at the border?? "staart"??
    heatmapsize = int(dispsize[1] + 2*strt), int(dispsize[0] + 2*strt)
    heatmap = np.zeros(heatmapsize, dtype=float)

    # create heatmap
    for xi,yi,duri in zip(x,y,dur):
        heatmap[yi:yi+gwh,xi:xi+gwh] += gaus * duri / 1000000
        
    heatmap = heatmap[strt:dispsize[1]+strt,strt:dispsize[0]+strt]
    
    return heatmap

def plot_heatmap(heatmap, dispsize, ax = None, alpha = .5, cmap = 'jet', remove_zeros = True,
                draw_numbers = False, S = None, T = None):
    
    hm_tmp = heatmap.copy()
    if remove_zeros:
        lowbound = np.mean(hm_tmp[hm_tmp>0]) # Remove points with below-mean 'heat'
        hm_tmp[hm_tmp<lowbound] = np.NaN
        
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=[8,5])
        ax.set(xlim = [0,1680], ylim = [0,1050], aspect = 1)
        
    ax.imshow(np.flipud(hm_tmp), cmap = cmap, alpha=alpha, zorder = 1)
    ax.set(xticks = [], yticks = [])     
        
    if draw_numbers:
        number_locations = get_number_locations()
#         print(number_locations)
        if S is not None:
            number_locations.loc[number_locations['num']=='S','num'] = S
        if T is not None:
            number_locations.loc[number_locations['num']=='T','num'] = T
        for ri, row in number_locations.iterrows():
            ax.text(row['X'],dispsize[1] - row['Y'],row['num'], fontdict = {'fontsize':15,
                'verticalalignment':'center','horizontalalignment':'center',
                'color':'r'}, zorder = 2)
    
#     ax.invert_yaxis()
    
    return ax

def load_block_pt(sub, session = None):
    baseDir = '/Users/jvanbaar/Dropbox (Brown)/Postdoc FHL/JEROEN/SOC_STRUCT_LEARN'
    
    if (session is None):
        filename_pattern = (baseDir +
            '/Study2_EyeTracking/Data/Raw/sub_%03d/ssldat_sub-%03d_ses-*.csv'%(
                sub, sub))
        filename = glob.glob(filename_pattern)[0]
#         print('Loading file%s:'%('s' if len(filenames) > 1 else ''))
#         [print(filenames[i]) for i in range(len(filenames))]
#         sessions = [int(re.findall('ses-\d\d\d',filenames[i])[0][-3:]) 
#                     for i in range(len(filenames))]
#         parts = [int(re.findall('part\d',filenames[i])[0][-1:]) 
#                     for i in range(len(filenames))]
        
    else:
        filename = (baseDir +
            '/Study2_EyeTracking/Data/Raw/sub_%03d/ssldat_sub-%03d_ses-%03d.csv'%(
            sub, sub, session))
    
    print('Loading file %s'%filename)
    game_dat = pd.read_csv(filename).dropna()
    game_dat['trial'] = game_dat['trial'].astype(int)
    
    game_dat.loc[game_dat['trial']==1,'block'] = np.arange(8)+1
    game_dat['block'].interpolate(method='pad',inplace=True)
    
    block_pt = game_dat[['player_type','block']].drop_duplicates().copy()
    block_pt['sub'] = sub
    block_pt = block_pt[['sub','block','player_type']].reset_index(drop=True)
    
    return block_pt

def get_heatmap_raw(trial_data, dispsize):
    
    ### GIVES VERY SIMILAR BUT NOT IDENTICAL RESULT TO FIXATION-BASED HEATMAP
    
    gwh = 200
    gsdwh = gwh/6
    gaus = gaussian(gwh,gsdwh)
    # matrix of zeroes
    strt = int(gwh/2) # This provides padding on all sides of the heatmap, for fixations at the border?? "staart"??
    heatmapsize = int(dispsize[1] + 2*strt), int(dispsize[0] + 2*strt)
    heatmap = np.zeros(heatmapsize, dtype=float)

    # create heatmap
    for i in range(len(trial_data)):
        # get x and y coordinates
        #x and y - indexes of heatmap array. must be integers
        x = int(round(trial_data['X'][i]))
        y = int(round(trial_data['Y'][i]))
        # only include fixation if it falls within display boundaries
        if (0 <= x <= dispsize[0]) and (0 <= y <= dispsize[1]):
            # add Gaussian to the current heatmap
            heatmap[y:y+gwh,x:x+gwh] += gaus
    # crop heatmap to display
    heatmap = heatmap[strt:dispsize[1]+strt,strt:dispsize[0]+strt]
    
    return heatmap