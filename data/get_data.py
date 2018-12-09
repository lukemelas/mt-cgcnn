import csv, os, json, math, sys, argparse
from pymatgen import MPRester
import numpy as np


class Logger(object):
    """Writes both to file and terminal"""
    def __init__(self, savepath, mode="a"):
        self.terminal = sys.stdout
        self.log = open(savepath + "logfile.log", mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self): # needed for python 3
        pass

def checkNull(dummy_dictionary):
    """Given a dictionary, checks if any of the values is null"""
    return None in list(dummy_dictionary.values())


def queryMPDatabse(api_key, input_filepath, properties, sample_fraction=1, fetch_cif=True, fetch_mpid=True, print_data=False):
    """
    Query the Materials Project Database

    Parameters
    ----------

    input_filepath: str
            Filepath containing list of material ids to be queried.
    properties: list
            The properties to query from the database e.g. ["formation_energy_per_atom", "band_gap"]
            NOTE: these are the properties we're trying to model the code to predict. For cif file or 
            material_id, use the other parameters below.
    sample_fraction: float
            After reading the input_filepath, sample_fraction of datapoints are selected and data for these
            samples are fetched from the database.
    fetch_cif: bool
            If true, fetches the cif data for the material
    fetch_mpid: bool
            if true, fetches the material_id for the material
    print_data: bool
            Print data in nice json format

    Returns
    -------

    dataset:
            Data corresponding to the query
    """
    materials_id_list = []
    with open(input_filepath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
                materials_id_list.append(str(row[0]))
    materials_id_list = np.random.choice(np.array(materials_id_list),\
        math.ceil(len(materials_id_list)*sample_fraction), replace=False).tolist()
    with MPRester(api_key) as m:
        print("Querying Materials Project Database...")
        query = {"material_id": {"$in": materials_id_list}}
        if fetch_cif:
            properties.append("cif")
        if fetch_mpid:
            properties.append("material_id")
        dataset = m.query(query, properties)
        if print_data:
            print(json.dumps(dataset, sort_keys=True, indent=4, separators=(',', ': ')))
        print("Done Querying. Fetched data for ", str(len(dataset)), " crystals")

    return dataset

def processData(dataset, output_filepath, has_cif=True, has_mpid=True, total_size=None):
    """
    Process data, filter out data with any a missing property,
    write data in files. 
    
    Parameters
    ----------
    dataset:
            Dataset containing the fetched data stored rowwise
    has_cif: bool
            If true, dataset contains the cif data for the material
    has_mpid: bool
            If true, dataset contains the material_id for the material
    total_size: int
            The maximum size of the dataset to be created
    """

    print("Processing dataset...")
    material_hash_counter = 0       # unique counter to identify each material
    material_id_hash_list = []
    idprop_list = []
    cif_list = []

    # Hack for the case when total_size is None
    if total_size is None:
        total_size = len(dataset)

    for row in dataset:
        property_keys = [*row]                  # list of keys
        property_values = list(row.values())    # list of values
        missing_data = True if None in property_values else False
        if not missing_data and total_size > 0:
            if has_cif:
                cif = row["cif"]
                cif_list.append([material_hash_counter, cif])
                property_keys.remove("cif")
            if has_mpid:
                material_id = row["material_id"]
                material_id_hash_list.append([material_hash_counter, material_id])
                property_keys.remove("material_id")
            fetched_properties = [row[x] for x in property_keys]
            idprop_list.append([material_hash_counter] + fetched_properties)
            total_size -= 1
            material_hash_counter += 1
    print("Finished processing dataset!")
    writeData(idprop_list, cif_list, material_id_hash_list, output_filepath)

def writeData(idprop_list, cif_list, material_id_hash_list, output_filepath):
    """
    Write data to files in a specific required structure format

    Parameters
    ----------
    idprop_list: list
            List containing unique material id and the properties fetched (no None cases)
    cif_list: list
            List containing unique material id and the cif file
    material_id_hash_list: list
            List containing unique material id and the material_id from Materials Project DB
    """
    print("Writing data to files...")
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    # Write id_prop.csv
    with open(output_filepath + '/id_prop.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(idprop_list)
    print("id_prop.csv")

    # Write the cif files
    for row in cif_list:
        unique_id, cif = row
        with open(output_filepath + '/' + str(unique_id) + '.cif', 'w') as f:
            f.write(cif)
        print(str(unique_id) + ".cif")

    # Write materials id hash map:
    # This contains the Unique ID and the material_id (as obtained from original dataset)
    if len(material_id_hash_list):
        with open(output_filepath + '/material_id_hash.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(material_id_hash_list)
    print("material_id_hash.csv")
    print("Complete.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--no_cifs', action='store_true')
    parser.add_argument('--mp_ids', type=str, default='csvs/mpids_sample.csv')
    parser.add_argument('--output', type=str, default='sample')
    args = parser.parse_args()

    # Get API key from hidden file
    if args.api_key is None:
        with open('.api_key.txt', 'r') as f:
            args.api_key = f.read().replace('\n', '')

    # Get properties
    properties = ["formation_energy_per_atom", "band_gap"]
    
    # Query Materials Project
    dataset = queryMPDatabse(args.api_key,
                             args.mp_ids,
                             properties,
                             print_data=False, 
                             fetch_cif=not args.no_cifs, 
                             sample_fraction=1)

    # Write to file
    processData(dataset, args.output, has_cif=not args.no_cifs)
