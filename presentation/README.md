A LaTeX template for a beamer of size 16:9. The template consist of a KU-folder with imagefiles, a KUstyle.sty file and a main.tex file.

IMPORTANT! In order for the document to compile, one needs to use XeLaTeX or LuaLaTeX as compiler. This can be done in  Overleaf by Menu -> Settings -> Compiler -> Choose XeLaTeX/LuaLaTeX

To change the text at the top, then change the 'toplinje' command in the document-enviromnet.

The template consist of a frontpage and a standard slide. A new slide can be added by using the frame-enviroment. It's important that [hoved] is included after \begin{frame}.

If one whises to print the beamer st. it exand to A4 size, one can add the following code before \begin{document}

\usepackage{pgfpages}
\pgfpagesuselayout{resize to}[a4paper,landscape,border shrink=5mm]