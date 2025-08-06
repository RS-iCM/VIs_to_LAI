import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

### --- Sub-function to read input & output option file ---
def read_io_opt(file_path):
    with open(file_path, 'r') as file:
        header1 = file.readline().strip()  # Read first line header
        n_yr = int(file.readline().strip())  # number of years
        
        yr = file.readline().strip()  # Read year
        sat_op = int(file.readline().strip())  # Satellite operation mode

        if sat_op == 1:  # high resolution RS data
            n_obs, min_obs = map(int, file.readline().split())
            doys = [file.readline().strip() for _ in range(n_obs)]
        elif sat_op == 2:  # low resolution RS data
            start_doy, last_doy, min_obs = map(int, file.readline().split())
            n_obs = last_doy - start_doy + 1 
            #print('n_obs:', n_obs)  ### TEST printing ###
            
            list_doys = list(range(start_doy, last_doy + 1, 8)) ### 8 day intervals ####
            #print('list_doys:', list_doys)  ### TEST printing ###
        
        cg_period = int(file.readline().strip())  # crop growth period
        area_n = file.readline().strip()  # area name
        n_vi = int(file.readline().strip())  # number of VIs
        t_vis = [file.readline().strip() for _ in range(n_vi)]  # VI names

        rows, cols = map(int, file.readline().split())  # read row and column numbers
        nmaxpixels = int(file.readline().strip())  # max pixels
        n_map_val = int(file.readline().strip())  # map projection number

        header2 = file.readline().strip().split()
        header3 = file.readline().strip().split()
        para_a1 = float(file.readline().strip())
        para_b1 = float(file.readline().strip())
        para_a2 = float(file.readline().strip())
        para_b2 = float(file.readline().strip())
        para_a3 = float(file.readline().strip())
        para_b3 = float(file.readline().strip())
        para_a4 = float(file.readline().strip())
        para_b4 = float(file.readline().strip())

    return {
        'n_yr': n_yr, 'yr': yr, 'sat_op': sat_op, 'n_obs': n_obs, 'min_obs': min_obs, 'list_doys': list_doys, 
        'start_doy': start_doy, 'last_doy': last_doy, 'cg_period': cg_period, 'area_n': area_n, 'n_vi': n_vi, 
        't_vis': t_vis, 'rows': rows, 'cols': cols, 'nmaxpixels': nmaxpixels, 'n_map_val': n_map_val,
        'params': {
            'para_a1': para_a1, 'para_b1': para_b1, 'para_a2': para_a2, 'para_b2': para_b2,
            'para_a3': para_a3, 'para_b3': para_b3, 'para_a4': para_a4, 'para_b4': para_b4
        }
    }

def read_data(workDir, filename, ncols): ## 2024.6.8
    file = workDir + '/'+filename
    
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f'Argument FILE {filename} is undefined')
        return None, 0
    
    nobs = len(lines)  # number of observations
    #print( 'nobs: \n', nobs)
    
    data = np.zeros((nobs, ncols))
    
    for i, line in enumerate(lines):
        record = np.fromstring(line, sep=' ')
        data[i, :] = record
    
    return data, nobs

def read_binary_data(file_path, dtype, count, offset):
    """Read binary data from a file with specified type, count, and offset."""
    with open(file_path, 'rb') as file:
        file.seek(offset)  # Move to the starting point in the file
        return np.fromfile(file, dtype=dtype, count=count)

def process_paddyFields_n_plantingDate_data(paddy_data_dir, yr, area_n, n_pixels, data_start):
    file_name_pFields = os.path.join(paddy_data_dir, f'asia.rice.extent.{yr}.{area_n}.dat')
    file_name_plantingDate = os.path.join(paddy_data_dir, f'transp.doy.{yr}.{area_n}.dat')

    sub_paddyFields = read_binary_data(file_name_pFields, dtype=np.int16, count=n_pixels, offset=data_start * 2)
    sub_plantingDate = read_binary_data(file_name_plantingDate, dtype=np.int16, count=n_pixels, offset=data_start * 2)

    sub_paddyFields = sub_paddyFields.astype(float)
    sub_plantingDate = sub_plantingDate.astype(float)

    sub_paddyFields[sub_paddyFields == -999] = np.nan
    sub_plantingDate[sub_plantingDate == -999] = np.nan

    return sub_paddyFields, sub_plantingDate

def read_spatial_VIs(data_dir, sat_opt, area_n, yr, list_doys, f_vis, n_obs, n_pixels, data_start):
    # Initialize data arrays for each VI and DOY data
    d_ndvi = np.zeros((n_obs, n_pixels))
    d_rdvi = np.zeros((n_obs, n_pixels))
    d_osavi = np.zeros((n_obs, n_pixels))
    d_mtvi1 = np.zeros((n_obs, n_pixels))
    rs_subdata = np.zeros((5, n_obs, n_pixels))
    data_doy = np.zeros(n_obs, dtype=int)

    fname_VIs_base = os.path.join(data_dir, 'MOD09A1.filtered')

    for i in range(0, n_obs, 8):
        if sat_opt == 1:
            yr_num = list_doys[i]
        else:
            yr_num = list_doys[0] + i  ### startDOY + i #print('yr_num:', yr_num) #### TEST print ###

        # Read each type of VI data
        for j, vi in enumerate(f_vis):
            file_name = f"{fname_VIs_base}{vi}{yr}{yr_num:03d}.{area_n}.dat"
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, 'rb') as file:
                file.seek(data_start * 2)  # Adjust according to how data_start is interpreted
                data = np.fromfile(file, dtype=np.int16, count=n_pixels) * 0.0001  # Apply scale factor
                
                # Check if the data array length matches the expected n_pixels
                if data.shape[0] != n_pixels:
                    raise ValueError(f"Expected number of pixels: {n_pixels}, but got: {data.shape[0]} from file: {file_path}")

            if j == 0:
                d_ndvi[i, :] = data
            elif j == 1:
                d_rdvi[i, :] = data
            elif j == 2:
                d_mtvi1[i, :] = data
            elif j == 3:
                d_osavi[i, :] = data
        
        data_doy[i] = yr_num 
    
    # assign to rs_subdata
    rs_subdata[0, :, :] = data_doy[:, None]  # Broadcast to all pixels
    rs_subdata[1, :, :] = d_ndvi
    rs_subdata[2, :, :] = d_rdvi
    rs_subdata[3, :, :] = d_mtvi1
    rs_subdata[4, :, :] = d_osavi

    return rs_subdata

def get_regr_coef_VI_LAI(data, n_VI): ###2024.6.8
    nlines = data.shape[0]
    X = np.ones((nlines, 3))
    X[:, 1] = np.log(data[:, 1])  # log(LAI)
    X[:, 2] = data[:, 0]          # DOY    
    Y = np.log(data[:, 2:n_VI + 2])    
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ Y           
    residuals = Y - X @ coef
    Sigma = residuals.T @ residuals / (nlines - 3)
    
    return coef, Sigma  

def get_LAI_from_VI(VI, yrNum, n_VI,coef,Sigma):
    X = np.array([1, yrNum])
    Sinv = np.linalg.inv(Sigma)
    # Handling negative numbers or zeros ##### 2024.08.20
    VI_safe = np.where(VI > 0, VI, 1e-10) ##### 2024.08.20
    res = np.log(VI_safe) - (coef[[0, 2],:].T @ X)   
    LAI = np.exp((coef[1,:] @ (Sinv @ res)) * (1.0/(coef[1,:].T @ (Sinv @ coef[1,:]))))
    
    return LAI

def get_VI_DOY(data, yrNum, n_VI,coef,Sigma):
    VI = np.zeros(n_VI)
    VI = data[0:n_VI]   
    LAI = get_LAI_from_VI(VI, yrNum, n_VI,coef,Sigma)
    
    return LAI

### -- Function to estimate LAI using VIs --
def log_VIs_to_log_LAI(n_obs,n_doys,n_pixels,sat_opt,start_doy,rs_data,paddyFields_data,plantingDate_data,
                       cg_period,n_vi,coef,Sigma):
    
    # Initialize arrays to hold LAI data
    LAI_data = np.full((n_obs, n_pixels), np.nan)
    sub_OLAI_data = np.full((n_doys, n_pixels), np.nan)
    
    # Store mean LAI and standard deviation
    sub_arr_mMLAI = np.full((3, n_doys), np.nan)  
    
    # Process data in eight-day intervals
    count = 0
    for i in range(0, n_obs, 8): 
        num2 = 0
        if sat_opt == 1:
            yrNum = n_doys[i]
        elif sat_opt == 2:
            yrNum = start_doy + i 

        for j in range(n_pixels):
            if rs_data[1, i, j] <= 0:
                LAI_data[i, j] = np.nan

            elif rs_data[1,i,j] > 0 and paddyFields_data[j] == 1 and yrNum > plantingDate_data[j] + 10 and \
            yrNum < plantingDate_data[j] + cg_period:
                
                LAI_value = get_VI_DOY(rs_data[1:n_vi+1, i, j], yrNum, n_vi,coef,Sigma)
                
                LAI_value_0 = LAI_value if i == 0 else 0.1
                if LAI_value_0 < 2.0 and LAI_value < 10.0 and LAI_value > 0.1:
                    LAI_data[i, j] = LAI_value

            num2 += 1  #if num2 % 2000 == 0: print(f'Run #{num2}')
        count += 1
        #if count % 5 == 0: print(f'Run #{count}')

        # Calculate mean LAI if there are enough valid values
        valid_data = LAI_data[i, ~np.isnan(LAI_data[i, :])]
        if valid_data.size > 1:  # Ensure there is more than one valid data point
            mMLAI = np.nanmean(valid_data)
            stdMLAI = np.nanstd(valid_data)
        else:
            mMLAI = np.nan  # Handle case where not enough valid values
            stdMLAI = np.nan
            
        if yrNum >= start_doy:
            sub_arr_mMLAI[0, count-1] = yrNum
            sub_arr_mMLAI[1, count-1] = mMLAI
            sub_arr_mMLAI[2, count-1] = stdMLAI
            
        sub_OLAI_data[count-1, :] =  LAI_data[i, :]

    return sub_OLAI_data, sub_arr_mMLAI
   
### -- Function to plot & write mean LAI --
def plot_n_write_meanLAI(output_dir, year, arr_mMLAI, plot_file_path = None):
    
    arr_mMLAI_transposed = arr_mMLAI.T  # Transpose the array
    # Convert the array to a pandas DataFrame
    df = pd.DataFrame(
        arr_mMLAI_transposed,
        columns=['DOY', 'Mean_LAI', 'STD_LAI'])

    # Plotting the mean LAI
    plt.figure(figsize=(7, 5))
    plt.errorbar(df['DOY'], df['Mean_LAI'], yerr=df['STD_LAI'], fmt='o', capsize=5)
    plt.title('Mean LAI over DOY')
    plt.xlabel('Day of Year (DOY)')
    plt.ylabel('Mean Leaf Area Index ($m^{2}$ $m^{-2}$)')
    if plot_file_path:
        directory = os.path.dirname(plot_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(plot_file_path)      
        print("Plot saved to:", plot_file_path)      
    plt.show()

    # Format each column individually
    df['DOY'] = df['DOY'].astype(int).map("{:3d}".format)  # Convert to int and then format as '3d'
    df['Mean_LAI'] = df['Mean_LAI'].map("{:.3f}".format)  # '.4f' for float
    df['STD_LAI'] = df['STD_LAI'].map("{:.3f}".format)  # '.4f' for float
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save DataFrame to a text file
    file_mLAI = os.path.join(output_dir, f"{year}_Mean_LAI.txt")

    # Save DataFrame to a text file with the formatted data
    df.to_csv(file_mLAI, sep='\t', index=False)  # Save as tab-separated values file
    print(f"Data saved to {file_mLAI}") ## Print confirmation that file was saved

### --- Function to write LAI data in a file ---
def write_out_LAI(output_dir, LAI_data, year, DOYs, sat_opt):
    # Create a directory path for the year if it doesn't exist
    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)  # Ensures that the directory exists

    # Write LAI data for each DOY to a file within the year directory
    for i, doy in enumerate(DOYs):
        file_LAI = os.path.join(year_dir, f"{year}{doy:03d}_LAI.bin")
        with open(file_LAI, 'wb') as file:
            np.array(LAI_data[i], dtype=np.float32).tofile(file)

### --- Function to map LAI data ---
def mapping_LAI(output_dir, year, DOY, map_dir, LAI_data, rows, cols, map_extent, image_extent, map_file_path=None):
    # Set the projection to rotate the map ##### 2024.08.18
    rotated_projection = ccrs.RotatedPole(pole_longitude=0, pole_latitude=88.24) ##### 2024.08.18
    image_projection = ccrs.PlateCarree() ##### 2024.08.18

    fig = plt.figure() ##### 2024.08.18

    # Default axis for drawing images (not rotated) ##### 2024.08.18
    ax_img = fig.add_subplot(1, 1, 1, projection=image_projection) ##### 2024.08.18
    ax_img.set_extent(map_extent, crs=image_projection) ##### 2024.08.18

    # Plot data with geographical extent
    if LAI_data.size == rows * cols:
        lai_img=ax_img.imshow(LAI_data.reshape((cols, rows)), origin='upper',
                            extent=image_extent, transform=image_projection, ##### 2024.08.17.                   
                            interpolation='nearest', cmap='viridis')
        cbar = plt.colorbar(lai_img, orientation='vertical',
                       fraction=0.046, pad=0.04, aspect=20, shrink=0.8)
        cbar.set_label('Leaf Area Index ($m^{2}$ $m^{-2}$)', fontsize=12, labelpad=5) # Set the title of the color bar
        
    else:
        raise ValueError("LAI_data size does not match the product of rows and cols")

    # Map overlay (using RotatedPole projection) ##### 2024.08.18
    ax_map = fig.add_subplot(1, 1, 1, projection=rotated_projection, label='map') ##### 2024.08.18
    ax_map.set_extent(map_extent, crs=image_projection) ##### 2024.08.18

    # Add continents or shape files from a directory
    shape_feature = ShapelyFeature(Reader(map_dir).geometries(),
                               ccrs.PlateCarree(), edgecolor='black')
    ax_map.add_feature(shape_feature, facecolor='none')

    # Make the background of ax_map transparent ##### 2024.08.18
    ax_map.patch.set_alpha(0) ##### 2024.08.18

    # Hide the default map frame (spines) ##### 2024.08.18
    for spine in ax_img.spines.values(): ##### 2024.08.18
        spine.set_visible(False) ##### 2024.08.18

    for spine in ax_map.spines.values(): ##### 2024.08.18
        spine.set_visible(False) ##### 2024.08.18

    # Matching the aspect (ratio) of two axes ##### 2024.08.18    
    ax_map.set_aspect(aspect='auto') ##### 2024.08.18    
    box_img = ax_img.get_position() ##### 2024.08.18  
    ax_map.set_position(box_img) ##### 2024.08.18  
        
    plt.title('LAI Mapping')       
    
    if map_file_path:
        directory = os.path.dirname(map_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(map_file_path)
        print(f"Map saved to {map_file_path}")
    plt.show()

def main(cwd, paddy_data_dir, output_dir, map_dir, fdata, IO_file_path):
    """cwd = os.getcwd()
    paddy_data_dir = cwd + '/paddy_data/'
    output_dir = cwd + '/py_LAI_log_to_log/'    
    map_dir = cwd + '/Shape_Paju_SK/Paju.shp' 
    fdata = 'VIs_LAI_Whole_rice.OBS'

    IO_file_path = 'info_Read_RS_VIs_n_Out_LAI_py.inp' """
    IO_data = read_io_opt(IO_file_path)
    
    data_start = 0  # The data pointed getting started.    
    sat_opt = IO_data['sat_op']
    area_n = IO_data['area_n']
    n_vi = IO_data['n_vi']
    yr = IO_data['yr']
    data_dir = f"{cwd}/vis_rice_Paju/{yr}/"  ### Vegetation index data directory
    start_doy = IO_data['start_doy']    
    list_doys = IO_data['list_doys'] ### 8 days intervel from start DOYs (105) ###
    n_doys = len(list_doys)
    
    n_obs = IO_data['n_obs']
    f_vis = IO_data['t_vis'] ### ['.ndvi.', '.rdvi.', '.mtv1.', '.osav.']
    rows = IO_data['rows']
    cols = IO_data['cols']
    n_pixels = rows*cols
    cg_period = IO_data['cg_period']    
    params = [
        IO_data['params']['para_a1'], IO_data['params']['para_b1'], IO_data['params']['para_a2'], 
        IO_data['params']['para_b2'],IO_data['params']['para_a3'], IO_data['params']['para_b3'], 
        IO_data['params']['para_a4'], IO_data['params']['para_b4']
    ]
    n_map_val = IO_data['n_map_val'] # number of the doys list for map projection
    nmaxpixels = IO_data['nmaxpixels']  # pixel numbers in the split run

    rs_data = np.zeros((5, n_obs, n_pixels))
    plantingDate_data = np.zeros(n_pixels, dtype=float)
    paddyFields_data = np.zeros(n_pixels, dtype=float)
    OLAI_data = np.full((n_doys, n_pixels), np.nan)
    arr_mMLAI = np.zeros((3, n_doys), dtype=float)
    
    split_op = 1  ### Option for splitting pixels: 0 = no split, 1 = splitting
    """ ### Calling without splitting ### """
    if split_op == 0:
        # calling 'read_spatial_VIs' function
        rs_data = read_spatial_VIs(data_dir,sat_opt,area_n,yr,list_doys,f_vis,n_obs,n_pixels,data_start)
        data, nlines = read_data(cwd, fdata, 2+n_vi)

        # calling 'process_paddyFields_n_plantingDate_data' function
        sub_paddyFields,sub_plantingDate = process_paddyFields_n_plantingDate_data(paddy_data_dir,yr,area_n,n_pixels, data_start)
  
        coef, Sigma = get_regr_coef_VI_LAI(data, n_vi) # 2024.6.8
  
        # calling 'empirical_log_VIs_to_log_lai' function
        sub_OLAI_data, sub_arr_mMLAI = log_VIs_to_log_LAI(n_obs,n_doys,n_pixels,sat_opt,start_doy,rs_data,
                                              sub_paddyFields,sub_plantingDate,cg_period,n_vi,coef,Sigma)
        OLAI_data = sub_OLAI_data
        arr_mMLAI = sub_arr_mMLAI
    """ ### END calling without splitting ### """

    """ ### Processing for each split to save memory ### """
    if split_op == 1:   
        nsplits = int(n_pixels/nmaxpixels)
        last_n_pixels = (1 * rows * cols) % nmaxpixels   
        for isplit in range(nsplits + 1):
            bFirstBlk = (isplit == 0)
            if isplit == nsplits:
                npixels = last_n_pixels
            else:
                npixels = nmaxpixels
            print(f'Run = {isplit} / {nsplits}')
            
            data_start = isplit*nmaxpixels

            # calling 'read_spatial_VIs' function
            sub_rs_data = read_spatial_VIs(data_dir,sat_opt,area_n,yr,list_doys,f_vis,n_obs,npixels,data_start)
            data, nlines = read_data(cwd, fdata, 2+n_vi)

            # calling 'process_paddyFields_n_plantingDate_data' function
            sub_paddyFields,sub_plantingDate = process_paddyFields_n_plantingDate_data(paddy_data_dir,yr,area_n,npixels, data_start)
  
            coef, Sigma = get_regr_coef_VI_LAI(data, n_vi) # 2024.6.8
  
            # calling 'empirical_log_VIs_to_log_lai' function
            sub_OLAI_data, sub_arr_mMLAI = log_VIs_to_log_LAI(n_obs,n_doys,npixels,sat_opt,start_doy,sub_rs_data,
                                              sub_paddyFields,sub_plantingDate,cg_period,n_vi,coef,Sigma)

            if nsplits == 0:
                rs_data = sub_rs_data
                paddyFields_data = sub_paddyFields
                plantingDate_data = sub_plantingDate
                OLAI_data = sub_OLAI_data
                arr_mMLAI = sub_arr_mMLAI

            if nsplits > 0:
                for i in range(npixels):
                    rs_data[:, :, isplit * nmaxpixels + i] = sub_rs_data[:, :, i]
                    plantingDate_data[isplit * nmaxpixels + i] = sub_plantingDate[i]
                    paddyFields_data[isplit * nmaxpixels + i] = sub_paddyFields[i]
                    OLAI_data[:, isplit * nmaxpixels + i] = sub_OLAI_data[:,i]
                    
                #arr_mMLAI = arr_mMLAI + sub_arr_mMLAI   # sum of the sub_mMLAI arrays
                arr_mMLAI += np.nan_to_num(sub_arr_mMLAI)  ### 2024.08.19
                
        arr_mMLAI = arr_mMLAI/nsplits   # averages of the total values
        arr_mMLAI[0, :n_doys] = list_doys  ### 2024.08.20
        arr_mMLAI[arr_mMLAI == 0] = np.nan  ### 2024.08.19
        
    """ ### END Processing for each split to save memory ### """
    
    # calling 'plot_n_write_meanLAI' function
    plot_output_path=f"{output_dir}/{yr}_Mean_LAI.png"  # in case of saving the map
    plot_n_write_meanLAI(output_dir, yr, arr_mMLAI, plot_file_path = None) ### plot_output_path)

    # Calling 'write_out_LAI' function
    write_out_LAI(output_dir, OLAI_data, yr, list_doys, sat_opt)

    # Calling 'mapping_LAI' function
    map_extent = [126.65, 127, 37.65, 38.0] ## -- [lon_min, lon_max, lat_min, lat_max
    image_extent = [126.66, 127.01, 37.69, 37.98] ## -- [lon_min, lon_max, lat_min, lat_max
    
    num = n_map_val   ## number of the doys list
    map_output_path=f"{output_dir}/{yr}.{list_doys[num]}_LAI_map.png"  # saving the map
    mapping_LAI(output_dir, yr, list_doys[num], map_dir, OLAI_data[num], rows, cols, map_extent,image_extent,
                map_file_path = None) ### map_output_path)

    if np.isnan(OLAI_data[num]).any():
        valid_mean = np.nanmean(OLAI_data)  # Calculate mean from the whole dataset or a broader dataset part
        print('mean:', "{:.3f}".format(valid_mean))

if __name__ == "__main__":
    # parse arguments or set defaults, then call main(...)
    pass