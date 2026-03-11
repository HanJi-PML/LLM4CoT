# LLM4CoT
This is the codes work Large Language Models for China-of-Task (LLM4CoT) Optimization

The description for all datasets is given as follows:
\begin{table}[t!]
\centering
\caption{Illustration of the dataset configuration of the different scenarios considered.}
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{|c|c|c|c|c|}
\hline
\multicolumn{2}{|c|}{\textbf{Scenarios}} & \textbf{Room size} & \textbf{AP numbers} & \textbf{UE numbers} \\ \hline
\multirow{12}{*}{\rotatebox{90}{Train}} & Room 1-1 (D1) & \multirow{2}{*}{5*5 $\text{m}^2$} & 5 & \multirow{4}{*}{[5, 10, 15, 20]} \\ \cline{2-2}\cline{4-4}
& Room 1-2 (D2) &  & 6 & \\ \cline{2-4}
& Room 2-1 (D3) & \multirow{2}{*}{6*6 $\text{m}^2$}  & 5 & \\ \cline{2-2}\cline{4-4}
& Room 2-2 (D4) &  & 7 & \\ \cline{2-5}
& Room 3-1 (D5) & \multirow{2}{*}{7*7 $\text{m}^2$}  & 6 & \multirow{4}{*}{[10, 15, 20, 30]} \\ \cline{2-2}\cline{4-4}
& Room 3-2 (D6) &  & 8 & \\ \cline{2-4}
& Room 4-1 (D7) & \multirow{2}{*}{8*8 $\text{m}^2$} & 7 &  \\ \cline{2-2}\cline{4-4}
& Room 4-2 (D8) &  & 9 &  \\ \cline{2-5}
& Room 5-1 (D9) & \multirow{2}{*}{9*9 $\text{m}^2$} & 8 & \multirow{4}{*}{[10, 20, 30, 40]}\\ \cline{2-2}\cline{4-4}
& Room 5-2 (D10) &  & 10 & \\ \cline{2-4}
& Room 6-1 (D11) & \multirow{2}{*}{10*10 $\text{m}^2$} & 9 &  \\ \cline{2-2}\cline{4-4}
& Room 6-2 (D12) &  & 10 & \\ \hline
\multirow{2}{*}{\rotatebox{90}{Test}} & Room 7 (D13) & 5*6 $\text{m}^2$ & 6 & \multirow{2}{*}{\makecell[c]{[5, 10, 15, 20, \\ 25, 30]}} \\ \cline{2-4}
& Room 8 (D14) & 5*8 $\text{m}^2$ & 7 &  \\ \hline
\end{tabular}
\label{Table:dataset}
\end{table}
