import csv

tex_file = open("table-rtcf-h.tex", "w")


table = r"""
\begin{table}
    \centering
    \caption{Convergence history for the hybridized Raviart-Thomas method of order $k$ on quadrilaterals ($k=0, \cdots, 4$).}
    \resizebox{\textwidth}{!}{%
        \begin{tabular}{l c c c c c c c}
            \toprule
            \multicolumn{8}{c}{RTCF-H method} \\
            \cmidrule{2-8}
            \multirow{2}{*}{$k$} & mesh &
            \multicolumn{2}{c}{$\norm{p-p_h}_{L^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} &
            \multicolumn{2}{c}{
                $\norm{\boldsymbol{u}-\boldsymbol{u}_h}_{\boldsymbol{L}^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} &
            \multicolumn{2}{c}{$\norm{p-p_h^{\star}}_{L^2(\Omega)} \leq \mathcal{O}(h^{k+2})$} \\
            & $r$ & error & order & error & order & error & order \\
            \bottomrule
            \multirow{5}{*}{0}
"""

with open("hybrid-mixed/H-RTCF-degree-0.csv") as csvfile0:
    readCSV = csv.reader(csvfile0, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 7):
                # Final column, need to insert 'next line' in table.
                if i == 6:
                    table += " & " + str(row[i]) + " \\\\ \n"
                else:
                    table += " & " + str(row[i])

table += r"""
\midrule
\multirow{5}{*}{1}
"""

with open("hybrid-mixed/H-RTCF-degree-1.csv") as csvfile1:
    readCSV = csv.reader(csvfile1, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 7):
                # Final column, need to insert 'next line' in table.
                if i == 6:
                    table += " & " + str(row[i]) + " \\\\ \n"
                else:
                    table += " & " + str(row[i])

table += r"""
\midrule
\multirow{5}{*}{2}
"""

with open("hybrid-mixed/H-RTCF-degree-2.csv") as csvfile2:
    readCSV = csv.reader(csvfile2, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 7):
                # Final column, need to insert 'next line' in table.
                if i == 6:
                    table += " & " + str(row[i]) + " \\\\ \n"
                else:
                    table += " & " + str(row[i])

table += r"""
\midrule
\multirow{5}{*}{3}
"""

with open("hybrid-mixed/H-RTCF-degree-3.csv") as csvfile3:
    readCSV = csv.reader(csvfile3, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 7):
                # Final column, need to insert 'next line' in table.
                if i == 6:
                    table += " & " + str(row[i]) + " \\\\ \n"
                else:
                    table += " & " + str(row[i])

table += r"""
\midrule
\multirow{5}{*}{4}
"""

with open("hybrid-mixed/H-RTCF-degree-4.csv") as csvfile4:
    readCSV = csv.reader(csvfile4, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 7):
                # Final column, need to insert 'next line' in table.
                if i == 6:
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
