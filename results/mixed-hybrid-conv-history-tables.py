import csv

tex_file = open("mixed-hybrid-tables.tex", "w")


# RT-method
tables = r"""
\begin{table}
    \centering
    \caption{Convergence history for the hybridized Raviart-Thomas method of order $k$ on simplices ($k=0, \cdots, 4$).}
    \resizebox{\textwidth}{!}{%
        \begin{tabular}{l c c c c c c c}
            \toprule
            \multicolumn{8}{c}{RT-H method} \\
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

with open("hybrid-mixed/H-RT-degree-0.csv") as csvfile0:
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
\midrule
\multirow{5}{*}{1}
"""

with open("hybrid-mixed/H-RT-degree-1.csv") as csvfile1:
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
\midrule
\multirow{5}{*}{2}
"""

with open("hybrid-mixed/H-RT-degree-2.csv") as csvfile2:
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
\midrule
\multirow{5}{*}{3}
"""

with open("hybrid-mixed/H-RT-degree-3.csv") as csvfile3:
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
\midrule
\multirow{5}{*}{4}
"""

with open("hybrid-mixed/H-RT-degree-4.csv") as csvfile4:
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
        \bottomrule
    \end{tabular}}
\end{table}
"""

# RTCF-method
tables += r"""
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
        \bottomrule
    \end{tabular}}
\end{table}
"""

tables += r"""
\begin{table}
    \centering
    \caption{Convergence history for the hybridized Brezzi-Douglas-Marini method of lowest order on simplices.}
    \resizebox{\textwidth}{!}{%
        \begin{tabular}{l c c c c c c c}
            \toprule
            \multicolumn{8}{c}{BDM-H method ($k=1$)} \\
            \cmidrule{2-8}
            \multirow{2}{*}{$k$} & mesh &
            \multicolumn{2}{c}{$\norm{p-p_h}_{L^2(\Omega)} \leq \mathcal{O}(h^{1})$} &
            \multicolumn{2}{c}{
                $\norm{\boldsymbol{u}-\boldsymbol{u}_h}_{\boldsymbol{L}^2(\Omega)} \leq \mathcal{O}(h^{2})$} &
            \multicolumn{2}{c}{$\norm{p-p_h^{\star}}_{L^2(\Omega)} \leq \mathcal{O}(h^{2})$} \\
            & $r$ & error & order & error & order & error & order \\
            \bottomrule
            \multirow{5}{*}{1}
"""

with open("hybrid-mixed/H-BDM-degree-1.csv") as csvfile1:
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
        \bottomrule
    \end{tabular}}
\end{table}
"""

tables += r"""
\begin{table}
    \centering
    \caption{Convergence history for the hybridized Brezzi-Douglas-Marini method of order $k$ on simplices ($k=2, 3, 4$).}
    \resizebox{\textwidth}{!}{%
        \begin{tabular}{l c c c c c c c}
            \toprule
            \multicolumn{8}{c}{BDM-H method} \\
            \cmidrule{2-8}
            \multirow{2}{*}{$k$} & mesh &
            \multicolumn{2}{c}{$\norm{p-p_h}_{L^2(\Omega)} \leq \mathcal{O}(h^{k})$} &
            \multicolumn{2}{c}{
                $\norm{\boldsymbol{u}-\boldsymbol{u}_h}_{\boldsymbol{L}^2(\Omega)} \leq \mathcal{O}(h^{k+1})$} &
            \multicolumn{2}{c}{$\norm{p-p_h^{\star}}_{L^2(\Omega)} \leq \mathcal{O}(h^{k+2})$} \\
            & $r$ & error & order & error & order & error & order \\
            \bottomrule
            \multirow{5}{*}{2}
"""

with open("hybrid-mixed/H-BDM-degree-2.csv") as csvfile2:
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
\midrule
\multirow{5}{*}{3}
"""

with open("hybrid-mixed/H-BDM-degree-3.csv") as csvfile3:
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
                    tables += " & " + str(row[i]) + " \\\\ \n"
                else:
                    tables += " & " + str(row[i])

tables += r"""
\midrule
\multirow{5}{*}{4}
"""

with open("hybrid-mixed/H-BDM-degree-4.csv") as csvfile4:
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
