import csv


tex_file = open("table-ldg-h-tau-order-h.tex", "w")


table = r"""
\begin{table}
    \centering
    \caption{Convergence history for the LDG-H method for $\tau = h$.}
    \resizebox{\textwidth}{!}{%
        \begin{tabular}{l c c c c c c c c c}
            \toprule
            \multicolumn{10}{c}{LDG-H method ($\tau = \mathcal{O}(h)$)} \\
            \cmidrule{2-10}
            \multirow{2}{*}{$k$} & mesh &
            \multicolumn{2}{c}{$\norm{p-p_h}_{L^2(\Omega)} \leq \mathcal{O}(h^{k})$} &
            \multicolumn{2}{c}{
                $\norm{\boldsymbol{u}-\boldsymbol{u}_h}_{\boldsymbol{L}^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} &
            \multicolumn{2}{c}{$\norm{p-p_h^{\star}}_{L^2(\Omega)} \leq \mathcal{O}(h^{k+2})$} &
            \multicolumn{2}{c}{
                $\norm{\boldsymbol{u}-\boldsymbol{u}_h^{\star}}_{\boldsymbol{L}^2(\Omega)} \leq \mathcal{O}(h^{k+1}) $} \\
            & $r$ & error & order & error & order & error & order & error & order \\
            \bottomrule
            \multirow{6}{*}{1}
"""

# Now we're ready to starting writing in row-by-row
# starting with degree = 1:
with open("LDG-H/LDG-H-d1-tau_order-h.csv") as csvfile0:
    readCSV = csv.reader(csvfile0, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 9):
                # Final column, need to insert 'next line' in table.
                if i == 8:
                    table += " & " + str(row[i]) + " \\\\ \n"
                else:
                    table += " & " + str(row[i])

# Now we're finished reading from this file. Start the next one and so on.
table += r"""
\midrule
\multirow{6}{*}{2}
"""
with open("LDG-H/LDG-H-d2-tau_order-h.csv") as csvfile1:
    readCSV = csv.reader(csvfile1, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 9):
                # Final column, need to insert 'next line' in table.
                if i == 8:
                    table += " & " + str(row[i]) + " \\\\ \n"
                else:
                    table += " & " + str(row[i])

table += r"""
\midrule
\multirow{6}{*}{3}
"""
with open("LDG-H/LDG-H-d3-tau_order-h.csv") as csvfile2:
    readCSV = csv.reader(csvfile2, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 9):
                # Final column, need to insert 'next line' in table.
                if i == 8:
                    table += " & " + str(row[i]) + " \\\\ \n"
                else:
                    table += " & " + str(row[i])

table += r"""
        \bottomrule
    \end{tabular}}
\end{table}
"""

tex_file.write(table)
tex_file.close()
