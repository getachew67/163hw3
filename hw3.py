"""
Name: Anastasia Stevens
Section: CSE 163 AC
hw3.py uses data on degree attainment in the United States from
the National Center for Education Statistics. It calculates statistics,
outputs several graphs, and fits a model to the data.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def parse(file):
    """
    Accepts a CSV file name and returns a DataFrame with the contents of
    the file. Assumes file exists. Assumes any missing values are marked with
    "---"
    """
    return pd.read_csv(file, na_values='---')


def completions_between_years(data, year1, year2, sex):
    """
    Accepts a non-empty DataFrame of education data, two years between 1990
    and 2018, and a sex (given by 'A' (all), 'F' (female), or 'M' (male)).
    Returns all rows of data which match the sex and are between the first
    year (inclusive) and the second year (exclusive). If no data is found
    for the parameters, returns None.
    """
    if (data is not None and not data.empty):
        of_sex = data['Sex'] == sex
        above_year1 = data['Year'] >= year1
        below_year2 = data['Year'] < year2
        selected = data[of_sex & above_year1 & below_year2]
        if len(selected) > 0:
            return selected
        else:
            return None


def compare_bachelors_1980(data):
    """
    Accepts a non-empty DataFrame of education data. Returns the percentages
    of women and men who earned a bachelor's degree in 1980, as a tuple, in
    the form: (% for men, % for women).
    """
    if (data is not None and not data.empty):
        in_1980 = data['Year'] == 1980
        bach_deg = data['Min degree'] == "bachelor's"
        subset = data[in_1980 & bach_deg]

        sex_male = subset[subset['Sex'] == 'M']['Total'].max()
        sex_female = subset[subset['Sex'] == 'F']['Total'].max()
        return sex_male, sex_female


def top_2_2000s(data):
    """
    Accepts a non-empty DataFrame of education data. Returns the two most
    commonly awarded levels of educational attainment between 2000 and 2010
    (inclusive) and their mean percentages over these years. Returns values as
    a tuple in the form: [(#1 level, mean % of #1 level), (#2 level, mean % of
    #2 level)].
    """
    if (data is not None and not data.empty):
        above_2000 = data['Year'] >= 2000
        below_2010 = data['Year'] <= 2010
        all_sex = data['Sex'] == 'A'

        subset = data[above_2000 & below_2010 & all_sex]
        subset = subset.groupby('Min degree')['Total'].mean()
        subset = subset.nlargest(2)

        result = list(zip(subset.index, subset))
        return result


def percent_change_bachelors_2000s(data, sex='A'):
    """
    Accepts a non-empty DataFrame of education data and sex ('M' indicating
    male, 'F' indicating female, and 'A' indicating all.) If a call does not
    specificy a sex, defaults to all sexes. Returns the difference between
    total percent of bachelor's degrees received in 2000 and 2010, as a float,
    for specified sex.
    """
    if (data is not None and not data.empty):
        bach_deg = data['Min degree'] == "bachelor's"
        of_sex = data['Sex'] == sex
        subset = data[of_sex & bach_deg]

        in_2000 = subset[subset['Year'] == 2000]['Total'].max()
        in_2010 = subset[subset['Year'] == 2010]['Total'].max()
        return in_2010 - in_2000


def line_plot_bachelors(data):
    """
    Accepts a non-empty DataFrame of education data. Graphs the total percent
    of both sexes to attain a bachelor's degree as minimal completion over
    the years in the dataset, as a line chart. Saves chart as:
    line_plot_bachelors.png
    """
    if (data is not None and not data.empty):
        all_sexes = data['Sex'] == 'A'
        min_bach = data['Min degree'] == "bachelor's"
        subset = data[all_sexes & min_bach]
        chart = sns.relplot(x='Year', y='Total', data=subset, kind='line')
        chart.set_axis_labels('Year', "Percentage of People with Minimum" +
                              "Bachelor's Degree")
        plt.savefig('line_plot_bachelors.png')


def bar_chart_high_school(data):
    """
    Accepts a non-empty DataFrame of education data. Graphs a bar chart of the
    total percentages of women, men, and all people with a minimum education
    of high school degrees in the year 2009. Saves bar chart as:
    bar_chart_high_school.png
    """
    if (data is not None and not data.empty):
        in_2009 = data['Year'] == 2009
        hs_ed = data['Min degree'] == 'high school'
        subset = data[in_2009 & hs_ed]
        chart = sns.catplot(x='Sex', y='Total', kind='bar', color='b',
                            data=subset)
        chart.set_xticklabels(['All Sexes', 'Male', 'Female'])
        chart.set_axis_labels('Sex', 'Percentage with Minimum High School'
                              + 'Degree')
        plt.savefig('bar_chart_high_school.png')


def plot_hispanic_min_degree(data):
    """
    Accepts a non-empty DataFrame of education data. Graphs a line chart
    showing the percentage of Hispanic individuals with minimum high school
    and bachelor's degrees between 1990 and 2010 (inclusive). Saves chart
    as: plot_hispanic_min_degree.png
    """
    if (data is not None and not data.empty):
        above_1990 = data['Year'] >= 1990
        below_2010 = data['Year'] <= 2010
        all_sexes = data['Sex'] == 'A'

        min_ed = (data['Min degree'] == "bachelor's") | (data['Min degree']
                                                         == "high school")

        subset = data[above_1990 & below_2010 & all_sexes & min_ed]
        chart = sns.relplot(x='Year', y='Hispanic', hue='Min degree',
                            data=subset, kind='line')
        chart.set_axis_labels('Year', 'Percentage of Hispanics with' +
                              'Education Level')
        sns.set_context("talk")
        plt.savefig('plot_hispanic_min_degree.png')


def fit_and_predict_degrees(data):
    """
    Accepts a non-empty DataFrame of education data. Uses a decision tree
    regressor model to calculate and return the test mean square error as a
    float.
    """
    if (data is not None and not data.empty):
        subset = data.loc[:, data.columns.intersection(['Year', 'Min degree',
                                                        'Sex', 'Total'])]
        subset = subset.dropna()
        subset = pd.get_dummies(subset)

        x = subset.drop(columns=['Total', 'Sex_A'])
        y = subset['Total']
        y = y.astype('float')

        X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.2)

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)

        return mean_squared_error(y_test, y_test_pred)


def main():
    file_name = 'hw3-nces-ed-attainment.csv'
    data = parse(file_name)
    # sns.set()
    # completions_between_years(data, 2007, 2008, 'F')
    # compare_bachelors_1980(data)
    print(top_2_2000s(data))
    # percent_change_bachelors_2000s(data, sex='A')
    # line_plot_bachelors(data)
    # bar_chart_high_school(data)
    # plot_hispanic_min_degree(data)
    # fit_and_predict_degrees(data)


if __name__ == "__main__":
    main()
