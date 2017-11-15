import csv

tex_file = open("ldg-h-tables.tex", "w")

# Start with the LDG-H method for tau = h
tables = r"""
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

# Now we're finished reading from this file. Start the next one and so on.
tables += r"""
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
        \bottomrule
    \end{tabular}}
\end{table}
"""

# Now for tau = O(1/h) (tau = 1/h in our examples)
tables += r"""
\begin{table}
    \centering
    \caption{Convergence history for the LDG-H method for $\tau = \frac{1}{h}$.}
    \resizebox{\textwidth}{!}{%
        \begin{tabular}{l c c c c c c c c c}
            \toprule
            \multicolumn{10}{c}{LDG-H method ($\tau = \mathcal{O}(h^{-1})$)} \\
            \cmidrule{2-10}
            \multirow{2}{*}{$k$} & mesh &
            \multicolumn{2}{c}{$\norm{p-p_h}_{L^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} &
            \multicolumn{2}{c}{
                $\norm{\boldsymbol{u}-\boldsymbol{u}_h}_{\boldsymbol{L}^2(\Omega)} \leq \mathcal{O}(h^{k})$} &
            \multicolumn{2}{c}{$\norm{p-p_h^{\star}}_{L^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} &
            \multicolumn{2}{c}{
                $\norm{\boldsymbol{u}-\boldsymbol{u}_h^{\star}}_{\boldsymbol{L}^2(\Omega)} \leq \mathcal{O}(h^{k})$} \\
            & $r$ & error & order & error & order & error & order & error & order \\
            \bottomrule
            \multirow{6}{*}{1}
"""

# Now we're ready to starting writing in row-by-row
# starting with degree = 1:
with open("LDG-H/LDG-H-d1-tau_order-hneg1.csv") as csvfile3:
    readCSV = csv.reader(csvfile3, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 9):
                # Final column, need to insert 'next line' in table.
                if i == 8:
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

# Now we're finished reading from this file. Start the next one and so on.
tables += r"""
\midrule
\multirow{6}{*}{2}
"""
with open("LDG-H/LDG-H-d2-tau_order-hneg1.csv") as csvfile4:
    readCSV = csv.reader(csvfile4, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 9):
                # Final column, need to insert 'next line' in table.
                if i == 8:
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
\midrule
\multirow{6}{*}{3}
"""
with open("LDG-H/LDG-H-d3-tau_order-hneg1.csv") as csvfile5:
    readCSV = csv.reader(csvfile5, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 9):
                # Final column, need to insert 'next line' in table.
                if i == 8:
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
        \bottomrule
    \end{tabular}}
\end{table}
"""

# And finally...the last example with tau = 1.
tables += r"""
\begin{table}
    \centering
    \caption{Convergence history for the LDG-H method for $\tau = 1$.}
    \resizebox{\textwidth}{!}{%
        \begin{tabular}{l c c c c c c c c c}
            \toprule
            \multicolumn{10}{c}{LDG-H method ($\tau = \mathcal{O}(1)$)} \\
            \cmidrule{2-10}
            \multirow{2}{*}{$k$} & mesh &
            \multicolumn{2}{c}{$\norm{p-p_h}_{L^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} &
            \multicolumn{2}{c}{
                $\norm{\boldsymbol{u}-\boldsymbol{u}_h}_{\boldsymbol{L}^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} &
            \multicolumn{2}{c}{$\norm{p-p_h^{\star}}_{L^2(\Omega)} \leq \mathcal{O}(h^{k+2})$} &
            \multicolumn{2}{c}{
                $\norm{\boldsymbol{u}-\boldsymbol{u}_h^{\star}}_{\boldsymbol{L}^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} \\
            & $r$ & error & order & error & order & error & order & error & order \\
            \bottomrule
            \multirow{6}{*}{1}
"""

# Now we're ready to starting writing in row-by-row
# starting with degree = 1:
with open("LDG-H/LDG-H-d1-tau_order-1.csv") as csvfile6:
    readCSV = csv.reader(csvfile6, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 9):
                # Final column, need to insert 'next line' in table.
                if i == 8:
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

# Now we're finished reading from this file. Start the next one and so on.
tables += r"""
\midrule
\multirow{6}{*}{2}
"""
with open("LDG-H/LDG-H-d2-tau_order-1.csv") as csvfile7:
    readCSV = csv.reader(csvfile7, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 9):
                # Final column, need to insert 'next line' in table.
                if i == 8:
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
\midrule
\multirow{6}{*}{3}
"""
with open("LDG-H/LDG-H-d3-tau_order-1.csv") as csvfile8:
    readCSV = csv.reader(csvfile8, delimiter=",")
    for j, row in enumerate(readCSV):
        # Ignore first row --- these are just column labels
        if j == 0:
            continue
        else:
            # Number of columns are labeled 0, ..., n-1.
            for i in range(0, 9):
                # Final column, need to insert 'next line' in table.
                if i == 8:
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
        \bottomrule
    \end{tabular}}
\end{table}
"""

tex_file.write(tables)
tex_file.close()
