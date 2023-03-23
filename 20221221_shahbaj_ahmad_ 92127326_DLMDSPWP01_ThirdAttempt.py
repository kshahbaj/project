# imports several libraries, including "sys", "threading", "create_engine" from "sqlalchemy',
# "pandas", "gridplot", "figure", and "show" from "bokeh.layouts", "bokeh.plotting", and
# "ColumnDataSource" from "bokeh.models", and "os".

import sys
# The "sys" module grants access to certain variables that are utilized or managed
# by the interpreter, as well as functions that have a strong interaction with the interpreter.

import threading
# "threading" provides a way to run multiple threads (concurrent units of execution)
# within a single process.

from sqlalchemy import create_engine
# "create_engine" is a function from the SQLAlchemy library that creates
# a connection to a database.

import pandas as pd
# "pandas" is a library for data manipulation and analysis.

from bokeh.layouts import gridplot
# "gridplot" is a function from Bokeh that allows multiple
# plots to be arranged in a grid layout.

from bokeh.plotting import figure, show
# "figure" is a function from Bokeh that creates a new figure for plotting.
# "show" is a function from Bokeh that displays a Bokeh object in the default
# output (e.g., a web browser).

from bokeh.models import ColumnDataSource
# "ColumnDataSource" is a class from Bokeh that creates a data source for a plot.

import os
# The "os" module offers a means of utilizing operating system-specific functionality,
# such as reading from or writing to the file system.


# Creating Data Base class so that we can create database connection.
# Hear DB -----> DataBase.
class DataBaseConnection:
    def __init__(self):
        self.thread_Local = threading.local()

    def getting_database(self, connecting_strings):
        connecting_p = getattr(self.thread_Local, 'connecting_p', {})
        connecting_s = connecting_strings
        try:
            if connecting_s in connecting_p and connecting_p[connecting_s] is not None:
                print("Existing connection id is : {0}".format(
                    id(connecting_p[connecting_s])))
                return connecting_p[connecting_s]
            else:
                reset_connection = self.reset_database_connection(
                    connecting_strings)
                return reset_connection
        except Exception as exception_1:
            exception_object = sys.exc_info()
            exception_type = exception_object[0]
            exception_table = exception_object[2]
            fm_name = os.path.split(
                exception_table.tb_frame.f_code.co_filename)[1]
            exception_string = str(
                exception_type) + " & " + str(fm_name) + " & " + str(exception_table.tb_lineno)
            print("Exception has been occurred in the data base connection as : {0}, at : {1}".format(
                exception_1, exception_string))

    def reset_database_connection(self, connecting_strings):
        connecting_new = None
        connecting_keypoint = connecting_strings
        connecting_p = getattr(self.thread_Local, 'connecting_p', {})
        try:
            if connecting_keypoint in connecting_p and connecting_p[connecting_keypoint] is not None:
                connecting_p[connecting_keypoint].dispose()
        except Exception as exception_2:
            exception_object = sys.exc_info()
            exception_type = exception_object[0]
            exception_table = exception_object[2]
            fm_name = os.path.split(
                exception_table.tb_frame.f_code.co_filename)[1]
            exception_string = str(
                exception_type) + " & " + str(fm_name) + " & " + str(exception_table.tb_lineno)
            print("Exception has been occurred in the data base connection as : {0}, at : {1}".format(
                exception_2, exception_string))
        for i in range(3):
            try:
                connecting_new = create_engine(connecting_strings)
                break
            except Exception as exception_3:
                exception_object = sys.exc_info()
                exception_type = exception_object[0]
                exception_table = exception_object[2]
                fm_name = os.path.split(
                    exception_table.tb_frame.f_code.co_filename)[1]
                exception_string = str(exception_type) + " & " + \
                    str(fm_name) + " & " + str(exception_table.tb_lineno)
                print("Exception has been occurred in the data base connection as : {0}, at : {1}".format(
                    exception_3, exception_string))
                connecting_new = None
                continue
        if connecting_new is not None:
            print("Creating the connection with HTML identity document : {0}".format(
                id(connecting_new)))
            connecting_p[connecting_keypoint] = connecting_new
            self.thread_Local.connecting_p = connecting_p
        return connecting_new


DB = DataBaseConnection()


# Function to choose which of the four ideal functions out of fifty ideal functions which fits-
# -the training datasets the best
def getting_idl_funct(trndata_funct, idldata_funct):

    # Changing the name of columns for the individuality of the dataset.
    trndata_funct.columns = trndata_funct.columns.str.replace('y', 'trn_y')
    idldata_funct.columns = idldata_funct.columns.str.replace('y', 'idl_y')

    # Using pandas to combine both tables depending on x
    mergingdata_funct = pd.merge(
        trndata_funct, idldata_funct, left_on='x', right_on='x', how='inner')

    # Constructing dataframes to hold our results in the table
    generating_idl_funct = pd.DataFrame()
    generating_maxiumum_deviation = pd.DataFrame()

    # Going through the combined data table in a loop
    for looping_, i in enumerate([colum for colum in mergingdata_funct.columns if 'trn_' in colum]):

        # Making a transient dataframe
        temporary_ = pd.DataFrame()
        for j in [colum for colum in mergingdata_funct.columns if 'idl_' in colum]:
            temporary_["{1}_leastsquare".format(i, j)] = (
                mergingdata_funct[i] - mergingdata_funct[j]) ** 2
        best_value = str(temporary_.sum().idxmin()).split("_")[1]
        generating_idl_funct[[best_value]
                             ] = mergingdata_funct[["idl_" + best_value]]

        # Obtaining the highest deviation
        generating_maxiumum_deviation[best_value] = [temporary_[
            "idl_" + best_value + "_leastsquare"].max() ** (1 / 2)]
    generating_idl_funct.insert(
        loc=0, column='x', value=mergingdata_funct['x'])
    return {'idl': generating_idl_funct, 'maximum': generating_maxiumum_deviation}


def tst_funct(tst_funct, idl_funct, maxiumum_deviation):
    # Merging test dataset with ideal function and created ideal functions tables
    mergingdata_funct = tst_funct.merge(idl_funct, on=['x'], how='left')
    tstdata_funct['idl_funct'] = None
    for _idl, ro in mergingdata_funct.iterrows():
        if_funct = None
        dlta_y_minimum = float('inf')
        for _p, _ro in maxiumum_deviation.T.iterrows():
            dlta_y_deviation = abs(ro['y'] - ro[_p])

            # we assign a funct, then the dlta(Δ) should not exceed-
            # -the max deviat by more then the factor of square root.
            if _ro[0] * (2 ** (1 / 2)) >= dlta_y_deviation and dlta_y_minimum > dlta_y_deviation:
                dlta_y_minimum = dlta_y_deviation
                if_funct = _p
        tstdata_funct.at[_idl, 'dlta_y_deviation'] = dlta_y_minimum if dlta_y_minimum < float(
            'inf') else None
        tstdata_funct.loc[_idl, 'Number_of_idl_funct'] = if_funct
        tstdata_funct.at[_idl,
                         'idl_y'] = mergingdata_funct[if_funct][_idl] if if_funct else None
    return tst_funct


# To create a function that can plot all different types of graphs.

def plot_graph(pgidl_dataframe, pgtrn_dataframe, pgdeviation_dataframe, pgtst_dataframe):

    # The training data function first graph with respect to first ideal function.
    plotgraph_1 = figure(
        title="The training data function first graph with respect to first ideal function.")
    plotgraph_1.diamond(pgidl_dataframe.x.to_list(), pgidl_dataframe[pgidl_dataframe.columns[1]].to_list(
    ), color='blue', legend_label='Idl data funct first')
    plotgraph_1.line(pgtrn_dataframe.x.to_list(), pgtrn_dataframe.trn_y1.to_list(
    ), color='gray', legend_label='Trning data funct first')

    # The training data function second graph with respect to second ideal function.
    plotgraph_2 = figure(
        title="The training data function second graph with respect to second ideal function.")
    plotgraph_2.diamond(pgidl_dataframe.x.to_list(), pgidl_dataframe[pgidl_dataframe.columns[2]].to_list(
    ), color='lime', legend_label='Idl data funct second')
    plotgraph_2.line(pgtrn_dataframe.x.to_list(), pgtrn_dataframe.trn_y2.to_list(
    ), color='gray', legend_label='Trning data funct second')

    # The training data function third graph with respect to third ideal function.
    plotgraph_3 = figure(
        title="The training data function third graph with respect to third ideal function.")
    plotgraph_3.diamond(pgidl_dataframe.x.to_list(), pgidl_dataframe[pgidl_dataframe.columns[3]].to_list(
    ), color='red', legend_label='Idl data funct third')
    plotgraph_3.line(pgtrn_dataframe.x.to_list(), pgtrn_dataframe.trn_y3.to_list(
    ), color='gray', legend_label='Trning data funct third')

    # The training data function fourth graph with respect to fourth ideal function.
    plotgraph_4 = figure(
        title="The training data function fourth graph with respect to fourth ideal function.")
    plotgraph_4.diamond(pgidl_dataframe.x.to_list(), pgidl_dataframe[pgidl_dataframe.columns[4]].to_list(
    ), color='yellow', legend_label='Idl data funct fourth')
    plotgraph_4.line(pgtrn_dataframe.x.to_list(), pgtrn_dataframe.trn_y4.to_list(
    ), color='gray', legend_label='Trning data funct fourth')

    # To create a function that compares a testing dataset to an ideal training dataset.
    plotgraph_5 = figure(title="Tsting dataset Vs Trning idl dataset funct")
    plotgraph_5.diamond(pgtst_dataframe.x.to_list(), pgtst_dataframe.y.to_list(
    ), color='violet', legend_label='Tst data funct')
    colour_lst = ('blue', 'lime', 'red', 'yellow')
    for f_, colum in enumerate(pgdeviation_dataframe.columns):
        ft_cpy = pgtst_dataframe.copy()
        ft_cpy.loc[pgtst_dataframe.Number_of_idl_funct !=
                   colum, 'idl_y'] = None
        plotgraph_5.diamond(pgtst_dataframe.x.to_list(), ft_cpy['idl_y'].to_list(
        ), color=colour_lst[f_], legend_label='Idl data funct {0}'.format(colum))

    # Graph that displays the maximum deviations of a dataset
    plotgraph_6 = figure(title="The Maximum Deviat")
    dy_df = pgdeviation_dataframe.values.tolist()[0]
    sorc = ColumnDataSource(
        data=dict(left=[1, 2, 3, 4], counts=dy_df, color=colour_lst))
    plotgraph_6.vbar(x='left', width=0.6, bottom=0.1, top='counts',
                     color='color', legend_field="counts", source=sorc)

    # Making a grid of graphs to get combined graphs
    griding = gridplot([[plotgraph_1], [plotgraph_2], [plotgraph_3], [plotgraph_4], [
                       plotgraph_5], [plotgraph_6]], width=1200, height=600)
    show(griding)


# Creating SQL database to store the resulting values in the SQL database.
class SQLLtUtl:
    def __init__(self, connecting_strings):
        self.connection = DB.getting_database(
            connecting_strings=connecting_strings)

    def putting_data_function(self, dataframe, table, connecting_strings):
        for i in range(0, 3):
            try:
                dataframe.to_sql(name=table, con=self.connection,
                                 index=False, if_exists='replace')
                break
            except Exception as exception_4:
                print("An exception has occurred as : {0}".format(exception_4))
                self.connection = DB.reset_database_connection(
                    connecting_strings=connecting_strings)
                continue


# connection to the DataBase to get all dataset and result values in the database table.
SQL_CONNECTING_STRG = "sqlite:///database.db"


if __name__ == "__main__":
    sqlt_utl = SQLLtUtl(connecting_strings=SQL_CONNECTING_STRG)

    # First database table :-  The database table containing the training data.
    trn_dataframe = pd.read_csv("train.csv")
    sqlt_utl.putting_data_function(
        dataframe=trn_dataframe, table='Training Dataset', connecting_strings=SQL_CONNECTING_STRG)

    # Second database table : -  The database table containing the ideal functions.
    idl_dataframe = pd.read_csv("ideal.csv")
    sqlt_utl.putting_data_function(
        dataframe=idl_dataframe, table='Ideal Dataset', connecting_strings=SQL_CONNECTING_STRG)

    # Third database table :-  The database table containing the test data.
    tstdata_funct = pd.read_csv("test.csv")
    sqlt_utl.putting_data_function(
        dataframe=tstdata_funct, table='Testing Dataset', connecting_strings=SQL_CONNECTING_STRG)

    # Obtaining four created ideal functions and maximum deviation.
    funct_dataframe = getting_idl_funct(trn_dataframe, idl_dataframe)

    # Mapping the test dataset with ideal function.
    tst_mapping_dataframe = tst_funct(
        tstdata_funct, funct_dataframe['idl'], funct_dataframe['maximum'])

    # Plotting the graph of mapping dataset.
    plot_graph(funct_dataframe['idl'], trn_dataframe,
               funct_dataframe['maximum'], tst_mapping_dataframe)

    # Fourth database table : -  The database table containing the test data, including the mapping and y-deviation.
    tst_mapping_dataframe = tst_mapping_dataframe[[
        'x', 'y', 'dlta_y_deviation', 'Number_of_idl_funct']]
    sqlt_utl.putting_data_function(dataframe=tst_mapping_dataframe,
                                   table='test_data_mapping_result', connecting_strings=SQL_CONNECTING_STRG)
