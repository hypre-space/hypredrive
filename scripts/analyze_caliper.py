#!/usr/bin/python

import argparse
import re
from anytree import Node, PreOrderIter
import pandas as pd
import plotly.express as px
import webbrowser

pd.set_option('display.max_rows', None)  # This line ensures all rows will be displayed

def read_caliper_data(filename):
    # List to store caliper data
    data = {"caliper_log": []}

    # Regex for finding total run time
    run_time_pattern = re.compile(r"run time.*\(([\d.]+) s\)")

    # Open file and Iterate through each line in the file
    found_header = False
    with open(filename, 'r') as fn:
        for line in fn:
            if not found_header:
                if line.startswith("Path"):
                    # Check if the line is the caliper header (Starting with "Path")
                    found_header = True
                    continue

                if match := run_time_pattern.search(line):
                    data["run_time"] = float(match.group(1))

            if found_header:
                if line.startswith("Function"):
                    # Exit if we reach the header of the MPI report section (starts with "Function")
                    found_header = False
                    continue

                # Process a line to replace spaces in the string column with "-"
                # Ensures we split from the right, keeping the first column intact if it has spaces
                parts = line.rsplit(maxsplit=4)

                # Find the index of the first non-space character
                first_non_space_index = next((index for index, char in enumerate(parts[0]) if char != ' '), len(parts[0]))

                # Keep the part before the first non-space character as is, replace spaces with hyphens in the rest
                parts[0] = parts[0][:first_non_space_index] + parts[0][first_non_space_index:].replace(' ', '-')

                # Reassemble the line
                data["caliper_log"].append(' '.join(parts))

    print(f"- Read {filename = }")
    return data

def create_trees(caliper_data, metric, verbose):
    rootname = 'Total'
    tree_labels = {0: Node(id = 0, name = rootname)}
    tree_timing = {0: Node(id = 0, name = rootname)}
    for myid, leaf in enumerate(caliper_data["caliper_log"]):
        # determine the node's depth
        leading_spaces = len(leaf) - len(leaf.lstrip(' '))
        level = int(leading_spaces / 2)

        # add the node to the stack, set as parent the node that's one level up
        split_caliper_line = re.split(' +',leaf.strip())
        pos_name = 0
        pos_time = 4
        clean_function_name = split_caliper_line[pos_name]
        if (len(split_caliper_line) < 4):
            continue
        if (verbose):
            print(split_caliper_line)
        region_time = split_caliper_line[pos_time]
        if metric in ('absolute', 'abs'):
            region_time = round(float(region_time) * caliper_data["run_time"] / 100.0, 1)

        elif metric in ('absolute-minutes', 'absm'):
            region_time = round(float(region_time) * caliper_data["run_time"] / 6000.0, 2)

        elif metric in ('absolute-hours', 'absh'):
            region_time = round(float(region_time) * caliper_data["run_time"] / 360000.0, 3)

        tree_labels[level + 1] = Node(id = myid + 1, name = clean_function_name, parent = tree_labels[level])
        tree_timing[level + 1] = Node(id = myid + 1, name = region_time, parent = tree_timing[level])
    largest_node_id = myid
    print(f"- Created 2 trees from Caliper log: one with labels, one with timings. They have {largest_node_id} nodes.")

    return tree_labels, tree_timing, largest_node_id

def get_all_branches(ftree):
    outlist = []
    max_length = -1
    all_branches = [list(leaf.path) for leaf in PreOrderIter(ftree, filter_=lambda node: node.is_leaf)]
    for this_branch in all_branches:
        bb = list(this_branch[-1].path)
        cc = [cc.name for cc in bb]
        outlist.append(cc)
        if len(cc) > max_length:
            max_length = len(cc)

    return outlist, max_length

def add_unaccounted_timing(label_tree, timing_tree, n_nodes, verbose):
    # Account for any part of the code with no Caliper timing (Core)

    # To detect unaccounted timing, we go through the tree, and for each
    # node with children, we sum the timing of children.
    # If the sum of children is less than the parent timing, we have unaccounted time.
    # In this case, we create a new child and assign all the unaccounted time to him.
    # If the sum of children is larger than the parent timing, which can happen due to roundoff error,
    # we subtract the difference from the last child.
    total_time_not_accounted_for = 0.
    label_when_not_counted='Core'
    myid = n_nodes
    for value_leaf, name_leaf in zip(PreOrderIter(timing_tree[0]), PreOrderIter(label_tree[0])):
        if value_leaf.id == 0:
            continue  # Skip because this is the tree Root
        if value_leaf.is_leaf:
            continue  # Skip leaves (leaves have no children)
        parent_value = float(value_leaf.name)

        # Check if all children add up to the parent value
        children_sum = 0
        for ch in value_leaf.children:
            children_sum += float(ch.name)
        difference = parent_value - children_sum

        # Add a node to both trees if sum of children is less than the parent value
        if difference > 1e-3:
            myid = myid + 1
            Node(id = myid, name = label_when_not_counted, parent = name_leaf)
            Node(id = myid, name = difference, parent = value_leaf)
            if verbose:
                print(f"- Added `Core` = {difference} to {name_leaf}")
            total_time_not_accounted_for += difference
        elif difference < 0:
            value_leaf.children[-1].name = str(float(value_leaf.children[-1].name) - difference)
            if verbose:
                print(f"- Corrected {name_leaf.children[-1]} by {difference}")

    print(f"- Added time not accounted for in Caliper (in total: {total_time_not_accounted_for}s)")
    return label_tree, timing_tree

def combine_trees_into_dataframe(label_tree, timing_tree):
    # Pandas magic happening here

    # Convert the label tree into a flat list of branches with their full paths (with only complete branches, going from root to leaf):
    aa, maxlength = get_all_branches(label_tree[0])

    # Create a default column name for each depth of the tree
    column_label = ['depth{}'.format(i) for i in range(maxlength)]
    # Turn the list of branches into a dataframe
    # The dataframe has as many row as complete branches
    # The dataframe has as many column as the longest branch
    # The shorter branchest are padded with "None" (eg. they look like: root depth1 depth2 leaf None None ... None)
    df = pd.DataFrame(aa, columns=column_label)

    # Do the same for the timing tree
    bb, maxlength = get_all_branches(timing_tree[0])
    timedf = pd.DataFrame(bb, columns=column_label)

    # Add timing information to the main dataframe
    # This is tricky: we need to find the right-most not-None column value for each row:
    df['timing'] = timedf.ffill(axis=1).iloc[:, -1].astype(float)

    # Then, we create a list of unique leaf labels
    leaf_labels = df.ffill(axis=1).iloc[:, -2]
    unique_labels = list(pd.unique(leaf_labels))

    # We assign an integer value to each unique label (here, we simply use the label index in unique_labels)
    df['label_id'] = leaf_labels.apply(lambda x: unique_labels.index(x))
    print('- Trees converted into a plottable DataFrame with the following columns: {}'.format(list(df.columns)))
    print('- The tree has {} unique labels.'.format(len(unique_labels)))
    return df, column_label

def main():
    metric_choices = ['percentage', 'absolute', 'abs', 'absh', 'absm']

    parser = argparse.ArgumentParser(description='Process caliper data.')
    parser.add_argument('-f', '--filename', required=True, help="Path to caliper's ASCII log")
    parser.add_argument('-o', '--output', default=None, help='Writes output file in HTML format')
    parser.add_argument('-t', '--title', default=None, help='Treemap title')
    parser.add_argument('-m', '--metric', default="percentage", choices=metric_choices, help='The metric type that is plotted')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    # Update title
    if args.title is None:
        args.title = f"Case: {args.filename.split('.')[0]}"

    # Data intake
    caliper_data = read_caliper_data(args.filename)
    tree_with_function_names, tree_with_runtime, n_nodes = create_trees(caliper_data, args.metric, args.verbose)
    tree_with_function_names, tree_with_runtime = add_unaccounted_timing(tree_with_function_names, tree_with_runtime, n_nodes, args.verbose)
    df, column_label = combine_trees_into_dataframe(tree_with_function_names, tree_with_runtime)
    if args.verbose:
        print(f"treemap {df = }")

    # Create an interactive Plotly Treemap diagram
    fig = px.treemap(df, path=column_label, values='timing')
    #fig = px.treemap(df, path=column_label, values='timing', color='label_id', color_continuous_scale='Rainbow')
    #fig.update_layout(autosize=False, width=1600, height=900)
    fig.data[0].hovertemplate = '%{label}<br>%{value}'
    fig.update_layout(
        title={
            'text': args.title,
            'y':0.99,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        title_font=dict(size=30)
    )
    #fig.update_coloraxes(showscale=False)

    # Save plotly diagram to output file and display html
    output_file = args.output + ".html" if (args.output) else args.filename.split(".")[0] + ".html"
    fig.write_html(output_file, include_plotlyjs="cdn")
    webbrowser.open(output_file)
    print(f"- Wrote output to {output_file}")
    print('- Done!')

if __name__ == "__main__":
    main()
