import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

#from RUN_Python_2D_LAI import maize_data_dir

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

def read_binary_data(file_path, dtype, count, offset):
    """Read binary data from a file with specified type, count, and offset."""
    with open(file_path, 'rb') as file:
        file.seek(offset)  # Move to the starting point in the file
        return np.fromfile(file, dtype=dtype, count=count)

# sub_Fields = process_Fields_n_plantingDate_data(paddy_data_dir,yr,area_n,npixels, data_start)
def process_Fields_n_plantingDate_data(maize_data_dir, yr, area_n, n_pixels, data_start, start_doy):
    file_name_pFields = os.path.join(maize_data_dir, f'class_map_{yr}.bin')
    #file_name_plantingDate = os.path.join(paddy_data_dir, f'transp.doy.{yr}.{area_n}.dat')

    sub_Fields = read_binary_data(file_name_pFields, dtype=np.int16, count=n_pixels, offset=data_start * 2)
    #sub_plantingDate = read_binary_data(file_name_plantingDate, dtype=np.int16, count=n_pixels, offset=data_start * 2)
    
    sub_Fields = sub_Fields.astype(float)
    #sub_plantingDate = sub_plantingDate.astype(float)

    sub_Fields[sub_Fields == -999] = np.nan
    #sub_plantingDate[sub_plantingDate == -999] = np.nan

    sub_plantingDate = np.full(n_pixels, start_doy - 11, dtype=float)

    return sub_Fields, sub_plantingDate

def read_spatial_VIs(data_dir, sat_opt, area_n, yr, list_doys, f_vis, n_obs, n_pixels, data_start):
    # Initialize data arrays for each VI and DOY data
    d_ndvi = np.zeros((n_obs, n_pixels))
    d_rdvi = np.zeros((n_obs, n_pixels))
    d_osavi = np.zeros((n_obs, n_pixels))
    d_mtvi1 = np.zeros((n_obs, n_pixels))
    rs_subdata = np.zeros((5, n_obs, n_pixels))
    data_doy = np.zeros(n_obs, dtype=int)

    fname_VIs_base = os.path.join(data_dir, 'MOD09A1.proj.utm')

    for i in range(0, n_obs, 8):
        if sat_opt == 1:
            yr_num = list_doys[i]
        else:
            yr_num = list_doys[0] + i  ### startDOY + i #print('yr_num:', yr_num) #### TEST print ###

        # Read each type of VI data
        for j, vi in enumerate(f_vis):
            file_name = f"{fname_VIs_base}{vi}interp.{yr}{yr_num:03d}.bin"
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

### -- Function to estimate LAI using VIs --
def empirical_VIs_to_lai(n_vi,n_obs,n_doys,n_pixels,sat_opt,start_doy,rs_data,paddyFields_data,
                          plantingDate_data,cg_period,params):
    
    # Unpack parameters for LAI calculations
    para_a1, para_b1, para_a2, para_b2, para_a3, para_b3, para_a4, para_b4 = params
    #print(para_a1, para_b1, para_a2, para_b2, para_a3, para_b3, para_a4, para_b4) ### TEST printing
    
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

            elif rs_data[1,i,j] > 0 and paddyFields_data[j] == 1 and yrNum > plantingDate_data[j] + 10 and yrNum < plantingDate_data[j] + cg_period:
                LAI_ndvi = para_a1 * np.exp(para_b1 * rs_data[1, i, j])
                LAI_rdvi = para_a2 * np.exp(para_b2 * rs_data[2, i, j])
                LAI_mtvi = para_a3 * np.exp(para_b3 * rs_data[3, i, j])             
                LAI_osavi = para_a4 * np.exp(para_b4 * rs_data[4, i, j])
                
                LAI_value = (LAI_ndvi + LAI_rdvi + LAI_mtvi + LAI_osavi) / 4.0
                LAI_value_0 = LAI_value if i == 0 else 0.1

                if LAI_value_0 < 2.0 and LAI_value < 10.0 and LAI_value > 0.1:
                    LAI_data[i, j] = LAI_value

            num2 += 1 
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
    #plt.title('Mean LAI over DOY')
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
    data_projection = ccrs.UTM(zone=15)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())  # type: ignore
    utm_extent = (203500.0, 734500.0, 4474000.0, 4818000.0)

    if LAI_data.size == rows * cols:
        lai_img = ax.imshow(LAI_data.reshape((cols, rows)), origin='upper',
                            extent=utm_extent, transform=data_projection,
                            interpolation='nearest', cmap='viridis')  # type: ignore
        try:
            ax.spines['geo'].set_visible(False)
        except AttributeError:
            ax.outline_patch.set_visible(False)  # type: ignore
        cbar = plt.colorbar(lai_img, orientation='vertical',
                       fraction=0.07, pad=0.04, aspect=15, shrink=0.5)
        cbar.set_label('Leaf Area Index ($m^{2}$ $m^{-2}$)', fontsize=12, labelpad=5)
    else:
        raise ValueError("LAI_data size does not match the product of rows and cols")
    shape_feature = ShapelyFeature(Reader(map_dir).geometries(),
                               data_projection, edgecolor='black')
    ax.add_feature(shape_feature, facecolor='none')  # type: ignore
    #plt.title('LAI Mapping')
    if map_file_path:
        directory = os.path.dirname(map_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(map_file_path)
        print(f"Map saved to {map_file_path}")
    plt.show()

def main(cwd, maize_data_dir, output_dir, map_dir, IO_file_path):

    IO_data = read_io_opt(IO_file_path)
    
    data_start = 0  # The data pointed getting started.    
    sat_opt = IO_data['sat_op']
    area_n = IO_data['area_n']
    n_vi = IO_data['n_vi']
    yr = IO_data['yr']
    data_dir = f"{cwd}/vis_maize_Iowa/{yr}/"  ### Vegetation index data directory
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
    Fields_data = np.zeros(n_pixels, dtype=float)
    OLAI_data = np.full((n_doys, n_pixels), np.nan)
    arr_mMLAI = np.zeros((3, n_doys), dtype=float)    
    
    split_op = 1 ### Option for splitting pixels: 0 = no split, 1 = splitting
    """ ### Calling without splitting ### """
    if split_op == 0:
        # calling 'read_spatial_VIs' function
        rs_data = read_spatial_VIs(data_dir, sat_opt, area_n, yr, list_doys, f_vis, n_obs, n_pixels, data_start)
    
        # calling 'process_paddyFields_n_plantingDate_data' function
        sub_Fields, sub_plantingDate = process_Fields_n_plantingDate_data(maize_data_dir,yr,area_n,n_pixels, data_start, start_doy)
    
        # calling 'empirical_VIs_to_lai' function
        sub_OLAI_data, sub_arr_mMLAI = empirical_VIs_to_lai(n_vi,n_obs,n_doys,n_pixels,sat_opt,start_doy,rs_data,sub_Fields,sub_plantingDate,cg_period,params)

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
            sub_rs_data = read_spatial_VIs(data_dir, sat_opt, area_n, yr, list_doys, f_vis, n_obs, 
                                       npixels, data_start)
    
            # calling 'process_Fields_n_plantingDate_data' function
            sub_Fields, sub_plantingDate = process_Fields_n_plantingDate_data(maize_data_dir, yr,area_n,npixels, data_start, start_doy)
    
            # calling 'empirical_VIs_to_lai' function
            sub_OLAI_data, sub_arr_mMLAI = empirical_VIs_to_lai(n_vi,n_obs,n_doys,
                                                        npixels,sat_opt,start_doy,sub_rs_data,
                                                        sub_Fields,sub_plantingDate,cg_period,params)    
            
            if nsplits == 0:
                rs_data = sub_rs_data
                Fields_data = sub_Fields
                plantingDate_data = sub_plantingDate
                OLAI_data = sub_OLAI_data
                arr_mMLAI = sub_arr_mMLAI

            if nsplits > 0:
                for i in range(npixels):
                    rs_data[:, :, isplit * nmaxpixels + i] = sub_rs_data[:, :, i]
                    plantingDate_data[isplit * nmaxpixels + i] = sub_plantingDate[i]
                    Fields_data[isplit * nmaxpixels + i] = sub_Fields[i]
                    OLAI_data[:, isplit * nmaxpixels + i] = sub_OLAI_data[:,i]
                    
                #arr_mMLAI = arr_mMLAI + sub_arr_mMLAI   # sum of the sub_mMLAI arrays
                arr_mMLAI += np.nan_to_num(sub_arr_mMLAI)  ### 2024.08.19
                
        arr_mMLAI = arr_mMLAI/nsplits   # averages of the total values
        arr_mMLAI[0, :n_doys] = list_doys  ### 2024.08.20
        arr_mMLAI[arr_mMLAI == 0] = np.nan  ### 2024.08.19
        
    """ ### END Processing for each split to save memory ### """

    # calling 'prot_n_write_meanLAI' function
    plot_output_path=f"{output_dir}/{yr}_Mean_LAI.png"  # in case of saving the map
    plot_n_write_meanLAI(output_dir, yr, arr_mMLAI, plot_file_path = None) ### plot_output_path)

    # Calling 'write_out_LAI' function
    write_out_LAI(output_dir, OLAI_data, yr, list_doys, sat_opt)

    # Calling 'mapping_LAI' function
    map_extent = [-96.7, -90.1, 40.3, 43.6] ## -- [lon_min, lon_max, lat_min, lat_max]
    image_extent = [-96.6, -90.2, 40.4, 43.5] ## -- [lon_min, lon_max, lat_min, lat_max]
    
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