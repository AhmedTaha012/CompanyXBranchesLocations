### Task Solution

### Apply Clustering Before Computing Estimated Sales

# 1-Cluster Locations:
#     Use a clustering algorithm OPTICS to group locations based on their geographic coordinates (latitude and longitude) with a distance parameter set to 5 km.
#     This will help identify clusters of locations that are within 5 km of each other.

# 2-Compute Combined Sales for Each Cluster:
#     For each cluster, sum the estimated sales of all locations within that cluster.
#     Assign the combined sales to a representative location (e.g., the centroid of the cluster or the location with the highest sales within the cluster).

# 3-Evaluate Clusters Against Operational Costs:
#     Determine if the combined sales for each cluster meet or exceed the operational costs.
#     Select the clusters (or representative locations) that achieve the best coverage and sales.

### Modules
from sklearn.cluster import OPTICS
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# Function to calculate distance matrix using Haversine formula
def haversine_matrix(latlon1, latlon2, lat2=None, lon2=None, bcase=True,single=False):
    """
    The `haversine_matrix` function calculates the haversine distance between pairs of latitude and
    longitude coordinates.
    
    :param latlon1: `latlon1` is a numpy array containing latitude and longitude coordinates for the
    first set of locations. The first column represents the latitude values, and the second column
    represents the longitude values
    :param latlon2: The `latlon2` parameter in the `haversine_matrix` function represents the latitude
    and longitude coordinates of the second location. It can be a single pair of coordinates or an array
    of coordinates if you are calculating distances between multiple points
    :param lat2: The `lat2` parameter in the `haversine_matrix` function represents the latitude
    coordinates of the second set of locations. It is used to calculate the distance between two sets of
    latitude and longitude coordinates on the Earth's surface using the Haversine formula
    :param lon2: The `lon2` parameter in the `haversine_matrix` function represents the longitude values
    for the second set of coordinates. It is used in conjunction with the `lat2` parameter to calculate
    the haversine distance between two sets of latitude and longitude coordinates
    :param bcase: The `bcase` parameter in the `haversine_matrix` function is a boolean flag that
    determines whether the input coordinates are provided as separate latitude and longitude arrays
    (`True`) or as pairs of latitude and longitude values (`False`), defaults to True (optional)
    :param single: The `single` parameter in the `haversine_matrix` function is a boolean flag that
    determines whether the function should calculate the haversine distance for a single pair of
    latitude and longitude coordinates or for multiple pairs of coordinates in matrix form, defaults to
    False (optional)
    :return: The function `haversine_matrix` returns the calculated distance between two sets of
    latitude and longitude coordinates in kilometers.
    """
    R = 6371  # Radius of Earth in kilometers
    if bcase:
        lat1, lon1 = latlon1[:, 0], latlon1[:, 1]
        lat2, lon2 = latlon2[:, 0], latlon2[:, 1]
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1[:, None]
        dlat = lat2 - lat1[:, None]
    else:
        lat1, lon1, lat2, lon2 = map(np.radians, [latlon1, latlon2, lat2, lon2])
        if single==True:
            delta_lat = lat2 - lat1
            delta_lon = lon2 - lon1
            a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c
        dlat = lat2 - lat1[:, np.newaxis]
        dlon = lon2 - lon1[:, np.newaxis]
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1[:, np.newaxis]) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def filterCluster(cluster_df):
    """
    The function `filterCluster` sorts points in a cluster by coverage, calculates Haversine distances
    from the point with maximum coverage to all other points, and returns the point that satisfies a
    condition or the point with the maximum coverage if no point meets the condition.
    
    :param cluster_df: The `filterCluster` function takes a DataFrame `cluster_df` as input, which
    presumably contains information about points in a cluster. The DataFrame is expected to have columns
    such as 'Coverage', 'Latitude', and 'Longitude' that are used in the function for filtering points
    based on certain criteria
    :return: The function `filterCluster` is returning either the point that satisfies the condition of
    having all distances within 5 km from the max coverage point in the cluster, or if no point
    satisfies the condition, it returns the point with the maximum coverage in the cluster.
    """
    # Sort points by Coverage in descending order
    sorted_points = cluster_df.sort_values(by='Coverage', ascending=False)
    for _, max_coverage_point in sorted_points.iterrows():
        # Calculate the Haversine distance from the max_coverage_point to all other points in the cluster
        distances = cluster_df.apply(lambda row: haversine_matrix(max_coverage_point['Latitude'], 
                                                           max_coverage_point['Longitude'], 
                                                           row['Latitude'], 
                                                           row['Longitude'],bcase=False,single=True), axis=1)
        
        # Check if all distances are within 5 km
        if all(distances <= 5):
            return max_coverage_point
    
    # If no point satisfies the condition, return MAX_COVER
    return sorted_points.iloc[0]

def selectPoints(df):
    """
    The function `selectPoints` groups points by cluster and applies a filtering function to each
    cluster.
    
    :param df: It seems like the code snippet you provided is incomplete. Could you please provide more
    information about the `filterCluster` function and the expected behavior of the `selectPoints`
    function? Additionally, could you provide a sample of the `df` dataframe that you are working with?
    This will help me better
    :return: The function `selectPoints` takes a DataFrame `df`, groups it by the 'cluster' column,
    applies the function `filterCluster` to each group, and then resets the index of the resulting
    DataFrame. The function returns the DataFrame containing the selected points after filtering by
    cluster.
    """
    selected_points = df.groupby('cluster').apply(filterCluster).reset_index(drop=True)
    return selected_points

# Filter rows based on distance
def filterRowsWithinDistance(df, distance_km=5):
    """
    The function `filterRowsWithinDistance` filters rows in a DataFrame based on their proximity to each
    other within a specified distance in kilometers using the haversine formula.
    
    :param df: A pandas DataFrame containing location data with columns 'Latitude' and 'Longitude'
    :param distance_km: The `distance_km` parameter in the `filterRowsWithinDistance` function
    represents the maximum distance in kilometers within which rows should be considered as within
    distance of each other. Any rows that are within this specified distance of each other will be
    filtered out from the final result, defaults to 5 (optional)
    :return: The function `filterRowsWithinDistance` returns a list of indices of rows in the input
    DataFrame `df` that are within a specified distance (default is 5 kilometers) of each other based on
    their latitude and longitude values.
    """
    filtered_indices = []
    for idx, row in df.iterrows():
        current_lat, current_lon = row['Latitude'], row['Longitude']
        within_distance = False
        for i in filtered_indices:
            lat, lon = df.loc[i, 'Latitude'], df.loc[i, 'Longitude']
            if haversine_matrix(current_lat, current_lon, lat, lon,bcase=False,single=True) <= distance_km:
                within_distance = True
                break
        if not within_distance:
            filtered_indices.append(idx)
    return filtered_indices

def plotDataFrameLongLatWithType(df,clstr=False,showdistance=False,showNumber=False):
    """
    The function `plotDataFrameLongLatWithType` creates a scatter plot of locations with options to
    display different types, clusters, distances, and numbers.
    
    :param df: The function `plotDataFrameLongLatWithType` takes a DataFrame `df` as input along with
    optional parameters `clstr`, `showdistance`, and `showNumber`
    :param clstr: The `clstr` parameter in the `plotDataFrameLongLatWithType` function is a boolean
    parameter that determines whether to display the scatter plot with different colors for each type
    and cluster. If `clstr` is set to `True`, the scatter plot will use different colors for each type
    and cluster, defaults to False (optional)
    :param showdistance: The `showdistance` parameter in the `plotDataFrameLongLatWithType` function is
    used to determine whether to display the distances between points on the scatter plot. If
    `showdistance=True`, the function will calculate the distances between all pairs of points and
    annotate these distances on the plot. The distances are, defaults to False (optional)
    :param showNumber: The `showNumber` parameter in the `plotDataFrameLongLatWithType` function is used
    to determine whether to display the cluster numbers on the scatter plot. If `showNumber=True`, the
    function will display the cluster numbers at the corresponding longitude and latitude positions on
    the plot. Each point on the plot, defaults to False (optional)
    """
    sns.set(style="whitegrid")

    # Create a figure and axis
    plt.figure(figsize=(12, 8))

    # Create a scatter plot with different colors for each type
    if clstr==True:
        sns.scatterplot(
        data=df,
        x='Longitude', y='Latitude',
        style="Type",hue='cluster',
        palette='Set3', s=100, alpha=0.7,legend='full'
        )
    else:
        sns.scatterplot(
        data=df,
        x='Longitude', y='Latitude',
        hue="Type",
        palette='Set3', s=100, alpha=0.7,legend='full'
        )

    # Set plot title and labels
    plt.title('Scatter Plot of Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if showdistance:
        # Calculate and annotate distances
        coords = df[['Latitude', 'Longitude']].to_numpy()
        for (i, j) in combinations(range(len(coords)), 2):
            dist = haversine_matrix(coords[i].reshape(1, -1), coords[j].reshape(1, -1))
            dist = dist.item()  # Extract scalar value from NumPy arra
            if dist <= 5:
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                plt.plot([lon1, lon2], [lat1, lat2], 'k-', alpha=0.3)
                mid_lat = (lat1 + lat2) / 2
                mid_lon = (lon1 + lon2) / 2
                plt.text(mid_lon, mid_lat, f'{dist:.2f} km', fontsize=9, ha='center', va='center')
    if showNumber:                
        for line in range(0, df.shape[0]):
            plt.text(df.Longitude[line], df.Latitude[line],
                    df.cluster[line], horizontalalignment='left', 
                    size=6, color='black', weight='semibold')
  

    # Show the plot
    plt.tight_layout()
    plt.savefig("./Assets/SavedLocatios.png")
    print("Result saved in Assets/SavedLocatios.png")



# The above code snippet is using the MinMaxScaler class from the scikit-learn library in Python. This
# class is typically used for scaling numerical features to a specified range, usually between 0 and
# 1. However, the code snippet provided is incomplete and lacks context, so it's difficult to
# determine the exact purpose or functionality without additional information.
srd=MinMaxScaler()

# The code is reading data from an Excel file that contains two sheets named 'Type A' and 'Type B'.
dataFrameTypeA=pd.read_excel("E:\Fawry_ML_Task\Data\Fawry - Data Science - AI Task.xlsx",sheet_name='Type A')
dataFrameTypeB=pd.read_excel("E:\Fawry_ML_Task\Data\Fawry - Data Science - AI Task.xlsx",sheet_name='Type B')

# The above code is adding a new column "Type" to two data frames, `dataFrameTypeA` and
# `dataFrameTypeB`, and assigning the values 'A' and 'B' respectively to indicate the type of each
# data frame. Additionally, it is adding a new column "operational_costs" to both data frames and
# assigning the values 25000 and 15000 to `dataFrameTypeA` and `dataFrameTypeB` respectively.
dataFrameTypeA["Type"]='A'
dataFrameTypeB["Type"]='B'
dataFrameTypeA["operational_costs"]=25000
dataFrameTypeB["operational_costs"]=15000

# concatenates the data from both sheets into a single DataFrame called `combinedDFOption2`.
# Finally, it assigns this concatenated DataFrame to another variable `combinedDfOPTICSOPT2`.
combinedDFOption2=pd.concat([dataFrameTypeA, dataFrameTypeB], ignore_index=True)
combinedDfOPTICSOPT2=combinedDFOption2

# The above code is creating an OPTICS clustering model with the following parameters:
# - `min_samples=2`: Minimum number of samples in a cluster
# - `eps=5`: The maximum distance between two samples for one to be considered as in the neighborhood
# of the other
# - `metric='haversine'`: The distance metric used for measuring the distance between points (in this
# case, haversine distance for geographical coordinates)
# - `algorithm="brute"`: The algorithm used to compute the nearest neighbors (brute force method)
optics = OPTICS(min_samples=2, eps=5, metric='haversine',algorithm="brute")
y_db = optics.fit_predict(srd.fit_transform(np.radians(combinedDfOPTICSOPT2[["Longitude","Latitude"]])))

# The code is adding a new column named 'cluster' to the DataFrame 'combinedDfOPTICSOPT2' and
# populating it with the values from output of OPTICS model.
combinedDfOPTICSOPT2['cluster'] = y_db

# The above code in Python is creating a new column 'updatedEstimatedSales' in the DataFrame
# 'combinedDfOPTICSOPT2'. This new column will contain the sum of 'Estimated_Sales' values for each
# group defined by the 'cluster' column. The 'transform' function is used to apply the sum operation
# within each group and assign the result to each row in the new column.
combinedDfOPTICSOPT2['updatedEstimastedSales'] = combinedDfOPTICSOPT2.groupby('cluster')['Estimated_Sales'].transform('sum')


# The code is filtering the DataFrame `combinedDfOPTICSOPT2` to only include rows where the value in
# the "updatedEstimastedSales" column is greater than the value in the "operational_costs" column.
combinedDfOPTICSOPT2=combinedDfOPTICSOPT2[combinedDfOPTICSOPT2["updatedEstimastedSales"]>combinedDfOPTICSOPT2["operational_costs"]]


# The above code is creating a new column called "Coverage" in the DataFrame `combinedDfOPTICSOPT2`.
# The values in this new column are calculated by subtracting the "operational_costs" column from the
# "updatedEstimatedSales" column. This calculation is done element-wise for each row in the DataFrame.
combinedDfOPTICSOPT2["Coverage"]=combinedDfOPTICSOPT2["updatedEstimastedSales"]-combinedDfOPTICSOPT2["operational_costs"]

# The code is filtering the `combinedDfOPTICSOPT2` DataFrame to only include rows where the value in
# the "cluster" column is equal to -1. The filtered data is then stored in the `noiseData` variable.
noiseData=combinedDfOPTICSOPT2[combinedDfOPTICSOPT2["cluster"]==-1]

# The code is finding the maximum value in the "cluster" column of the DataFrame
# `combinedDfOPTICSOPT2` where the value in the "cluster" column is not equal to -1. The result is
# stored in the variable `startcount`.
startcount=max(combinedDfOPTICSOPT2[combinedDfOPTICSOPT2["cluster"]!=-1]["cluster"])


# The code is creating a new column 'cluster' in the 'noiseData' DataFrame and populating it with a
# range of values starting from 'startcount' and ending at 'startcount-1+len(noiseData)+1'. Each row
# in the 'cluster' column will have a unique value within this range.
noiseData['cluster'] = range(startcount, (startcount-1)+len(noiseData) + 1)

# The code snippet is selecting points from a DataFrame by filtering out points with a cluster value
# of -1 and then concatenating the resulting DataFrame with another DataFrame called `noiseData`. The
# `selectPoints` function is then applied to the combined DataFrame to further process and select
# points.
selectedPoints = selectPoints(pd.concat([combinedDfOPTICSOPT2[combinedDfOPTICSOPT2["cluster"]!=-1], noiseData], ignore_index=True))

# The code is calling a function `filterRowsWithinDistance` with the argument `selectedPoints` and
# storing the result in the variable `indicesWithinDistance`. The function likely filters out rows
# from `selectedPoints` based on some distance criteria.
indicesWithinDistance=filterRowsWithinDistance(selectedPoints)

# The code is creating a new DataFrame called `resultDfOPTICSOPT2` by selecting rows from the
# DataFrame `selectedPoints` based on the indices specified in the variable `indicesWithinDistance`.
# The selected rows are then reset to have a new index starting from 0 using the
# `reset_index(drop=True)` method.
resultDfOPTICSOPT2 = selectedPoints.loc[indicesWithinDistance].reset_index(drop=True)

## Save image of result
plotDataFrameLongLatWithType(resultDfOPTICSOPT2,clstr=False,showdistance=True,showNumber=True)

## save excel sheet with locations
resultDfOPTICSOPT2.to_excel("selectedLocations.xlsx")


