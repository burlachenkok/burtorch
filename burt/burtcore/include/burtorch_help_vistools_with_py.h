#pragma once


#include <string>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <stdlib.h>

 /**
  * @brief Generates a Python command string to plot a function.
  *
  * This function creates a Python script as a command string that uses Matplotlib to plot
  * the values of a given mathematical function over a specified range.
  *
  * @tparam TArg The type of the argument values (e.g., double, float, int).
  * @tparam TFunc The function type that takes TArg as input and returns TRetType.
  * @tparam TRetType The return type of the function.
  * @param xStart The starting value of the x-axis range.
  * @param xEnd The ending value of the x-axis range.
  * @param func The mathematical function to be plotted.
  * @param functionName The name of the function (used in the plot title).
  * @param steps The number of discrete steps for sampling the function (default is 100).
  * @return A string containing a Python command that generates the plot.
  */
template <class TArg, class TFunc, class TRetType>
inline std::string generatePlot(TArg xStart, TArg xEnd, TFunc func, const char* functionName, size_t steps = 100)
{
    std::vector<TArg> x;
    std::vector<TRetType> y;

    for (size_t i = 0; i < steps; ++i)
    {
        TArg xi = xStart + (xEnd - xStart) / steps * i;
        TRetType yi = func(xi);

        x.push_back(xi);
        y.push_back(yi);
    }

    std::ostringstream s;
    s << "python -c \"";
    s << "import matplotlib.pyplot as plt" << ';';
    s << "x=[";
    for (size_t i = 0; i < steps; ++i)
    {
        s << x[i];
        if (i != steps - 1)
            s << ',';
    }
    s << ']' << ';';

    s << "y=[";
    for (size_t i = 0; i < steps; ++i)
    {
        s << y[i];
        if (i != steps - 1)
            s << ',';
    }
    s << ']' << ';';

    s << "plt.plot(x, y)" << ';';
    s << "plt.grid(True)" << ';';
    s << "plt.title('" << functionName << "');";
    s << "plt.show()" << ';';
    s << '"';
    return s.str();
}

/**
 * @brief Executes the generated Python command to display a function plot.
 *
 * This function calls `generatePlot` to create a Python command string, which is then
 * executed using the system command to display the function plot.
 *
 * @tparam TArg The type of the argument values (e.g., double, float, int).
 * @tparam TFunc The function type that takes TArg as input and returns a value.
 * @param xStart The starting value of the x-axis range.
 * @param xEnd The ending value of the x-axis range.
 * @param func The mathematical function to be plotted.
 * @param functionName The name of the function (used in the plot title).
 * @param steps The number of discrete steps for sampling the function (default is 100).
 */
template <class TArg, class TFunc>
inline void showPlot(TArg xStart, TArg xEnd, TFunc func, const char* functionName, size_t steps = 100)
{
    std::string cmd = generatePlot<TArg, TFunc, decltype(func(TArg()))>(xStart, xEnd, func, functionName, steps);
    system(cmd.c_str());
}

/**
 * Builds a DOT graph representation of a given tree-like structure.
 *
 * This function constructs a graph in the DOT format representing a tree structure. The graph is created
 * by traversing the nodes starting from the provided root node, and the relationship between each node and
 * its children is represented with directed edges.
 *
 * @tparam TValueType The type of the tree node, which should have methods to retrieve the node's index,
 *        label, help string, gradient, and data references, as well as access to its children.
 * @param root The root node from which the graph will be built.
 * @param graphName The name of the graph to be used in the DOT format.
 *
 * @return A string containing the DOT graph description.
 *
 * @note The graph uses the "digraph" format, and nodes are styled based on their type:
 *       - Root node: yellow, filled.
 *       - Leaf node (with no children): green, filled.
 *       - Other nodes: default style.
 *
 * @note This function uses depth-first search (DFS) for traversing the tree.
 */

template <class TValueType>
inline std::string buildDotGraph(const TValueType& root, std::string graphName) noexcept
{
    using TNodeIndexType = typename TValueType::TNodeIndexType;

    std::ostringstream connections_str;

    std::vector<TNodeIndexType> queue;
    std::unordered_set<TNodeIndexType> marked_nodes;
    std::unordered_map<TNodeIndexType, std::string> node_to_description;

    queue.push_back(root.sysGetRawNodeIndex());

    for (; !queue.empty();)
    {
        // Take item from DFS queue
        TNodeIndexType item = queue.back();
        queue.pop_back();

        // Already processed
        if (marked_nodes.find(item) != marked_nodes.end())
            continue;

        marked_nodes.insert(item);

        // Get children
        const auto& children = TValueType::sysViewMemoryAsNode(&item)->childrenSet();
        size_t numChidren = children.size();

        for (size_t i = 0; i < numChidren; ++i)
        {
            auto cvalue = children[i];
            if (marked_nodes.find(cvalue) == marked_nodes.end())
                queue.push_back(cvalue);
        }
        //========================================================================================//
        // process-node: comment is in that locally node and it has direct children that has indicies
        //========================================================================================//
        if (node_to_description.find(item) == node_to_description.end())
        {
            TValueType* itemNode = TValueType::sysViewMemoryAsNode(&item);
            std::ostringstream node_descr;
            node_descr << itemNode->getLabel() << '|';
            node_descr << itemNode->getHelpString();
            node_descr << "| grad:" << itemNode->gradRef();
            node_descr << "| data:" << itemNode->dataRef();
            node_descr << '|';
            node_descr << '#';
            node_descr << itemNode->sysGetRawNodeIndex();
            node_to_description[item] = node_descr.str();
        }

        for (size_t i = 0; i < numChidren; ++i)
        {
            auto cvalue = children[i];

            if (node_to_description.find(cvalue) == node_to_description.end())
            {
                TValueType* itemNode = TValueType::sysViewMemoryAsNode(&cvalue);
                std::ostringstream node_descr;
                node_descr << itemNode->getLabel() << '|';
                node_descr << itemNode->getHelpString();
                node_descr << "| grad:" << itemNode->gradRef();
                node_descr << "| data:" << itemNode->dataRef();
                node_descr << '|';
                node_descr << '#';
                node_descr << itemNode->sysGetRawNodeIndex();
                node_to_description[cvalue] = node_descr.str();
            }
            connections_str << item << "->" << cvalue << ";\n";
        }
    }

    // Print in dot graph format
    std::ostringstream s;
    s << "digraph \"" << graphName << "\" {" << "\n";
    s << "node [shape=record];\n";
    s << "\n";

    // structure binding from c++17
    for (auto [k, v] : node_to_description)
    {
        if (k == root.sysGetRawNodeIndex())
        {
            s << k << "[label = \"" << v << "\", style=filled, fillcolor=yellow];\n";
        }
        else if (TValueType::sysViewMemoryAsNode(&k)->childrenNum() == 0)
        {
            s << k << "[label = \"" << v << "\", style=filled, fillcolor=green];\n";
        }
        else
        {
            s << k << "[label = \"" << v << "\"];\n";
        }
    }

    s << "\n";
    s << connections_str.str();
    s << "}";
    s << "\n";

    return s.str();
}
