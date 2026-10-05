% crebss.cls -- For CREBSS journal published by CSA
% This class file is applicable for authors who must
% prepare an article for publication upon acceptance
% The user may not exploit any part of the class file commercially!
%
% This file is provided 'as it stands'. Authors are not permited
% to change this fclass ile.
%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{crebss}[2023 template for CREBSS]


\LoadClass[twoside,10pt]{article}

\RequirePackage[utf8]{inputenc}
\RequirePackage[english]{babel}  %\RequirePackage[croatian]{babel}
\RequirePackage[T1]{fontenc}
\RequirePackage[sc]{mathpazo}
\setlength{\parindent}{2em}
\renewcommand{\baselinestretch}{1.1}

\RequirePackage{amsfonts,amssymb,amsbsy,amsmath,amsthm}  
\RequirePackage{geometry} 
\RequirePackage{booktabs,caption}
\RequirePackage{subcaption}
\RequirePackage{float}
\RequirePackage{graphicx}
\RequirePackage{algorithmic}
\RequirePackage{algorithm}
\RequirePackage{epstopdf}
\RequirePackage{comment}
\RequirePackage{color}
\RequirePackage{xcolor}
\RequirePackage{tcolorbox}
\RequirePackage{mdframed}
\RequirePackage{fontawesome}
\RequirePackage[colorlinks=true,urlcolor=blue,citecolor=blue,linkcolor=blue]{hyperref}
\newcommand{\chdoilink}[2]{\href{http://dx.doi.org/{#1}{#2}}}
\RequirePackage{url}
\usepackage[noabbrev]{cleveref}
\RequirePackage{parcolumns}

\RequirePackage[authoryear,semicolon]{natbib}
\setlength{\bibsep}{0pt}
\bibliographystyle{plainnat}

\newboolean{PG@odd}  % set true for first page number is odd
\newboolean{REV@art} % set true for Review Article
\newboolean{DRAFT@copy} % set true for Draft copy
\newboolean{ONE@page} % set true for the first page end at abstract
\newboolean{START@even} % set true for the book first page is even
\setboolean{PG@odd}{true}
\setboolean{REV@art}{false}
\setboolean{DRAFT@copy}{false}
\setboolean{ONE@page}{false}
\setboolean{START@even}{false}
\DeclareOption{odd}{\setboolean{PG@odd}{true}}
\DeclareOption{even}{\setboolean{PG@odd}{false}}
\DeclareOption{review}{\setboolean{REV@art}{true}}
\DeclareOption{draft}{\setboolean{DRAFT@copy}{true}}
\DeclareOption{onepage}{\setboolean{ONE@page}{true}}
\DeclareOption{starteven}{\setboolean{START@even}{true}}

\ProcessOptions

% general setting
\headsep=5mm
\headheight=5mm
\topmargin=0pt
\oddsidemargin=0pt
\evensidemargin=0pt
%\textheight=195truemm   

\textheight=215truemm

%\textwidth=128truemm  

\textwidth=145truemm

\setlength{\arraycolsep}{1.7pt}

\setlength{\oddsidemargin}{8mm}
\setlength{\evensidemargin}{8mm}
\ifthenelse{\boolean{START@even}}
{ \setlength{\oddsidemargin}{8mm} 
  \setlength{\evensidemargin}{8mm} }
{}

\setlength\footnotesep{3.5\p@}
\footskip=8truemm

\renewcommand{\arraystretch}{1.0}
\abovedisplayskip=10pt plus 9pt minus 2.5pt%
\belowdisplayskip=10pt plus 9pt minus 2.5pt


\AtBeginDocument{
  \immediate\write\@auxout{\string
  \newlabel{FirstPage}{{}{\thepage}{}{Doc-Start}{}}}
}

\AtEndDocument{
  \clearpage\addtocounter{page}{-1}
  \immediate\write\@auxout{\string
  \newlabel{LastPage}{{}{\thepage}{}{}{}}}
  \addtocounter{page}{1}
}

\newcommand\journalname{Croatian Review of Economic, Business and Social Statistics}
\newcommand\thisnumber{2}
\newcommand\thisyear {2023}
\newcommand\thismonth{XXXX}
\newcommand\thisvolume{9}
\newcommand\datereceived{\today}
\newcommand\dateaccepted{\today}
\newcommand\dateavailable{\today}
\newcommand\doinumber{10.2478/crebss-2023-000x}
\newcommand\type{Review paper}
\newcommand\JEL{G73}


\def\thispagestart{\pageref{FirstPage}}
\def\thispageend{\pageref{LastPage}}


\def\ps@firstpage{\ps@empty  
\def\@oddfoot{\parbox{\textwidth}{
               \hfill \footnotesize \copyright \thisyear~Copyright of this article is retained by the author(s) \\ \null \hfill This is an open access article under the \href{https://creativecommons.org/licenses/by-nc-nd/4.0/}{CC BY--NC--ND 4.0} license} }   
  \let\@evenfoot\@oddfoot
%  \def\@oddhead{\usebox{\lheadbox} \hfil \usebox{\rheadbox}
  \def\@oddhead{
      \parbox{\textwidth}{\footnotesize Croatian Review of Economic, Business and Social Statistics \hfill{\normalsize\thepage}
\\ \textsc{CREBSS~\thisvolume(\thisnumber):\thispagestart--\thispageend}}
  }
  \let\@evenhead\@oddhead % in case an article starts on a left-hand page
  \let\@mkboth\@empty
}

\def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
%      \def\@oddhead{\footnotesize \thepage\hfil\leftmark\hfil}%
%      \def\@evenhead{\footnotesize \hfil\rightmark\hfil\thepage}%
\ifthenelse{\boolean{PG@odd}}
      { \def\@evenhead{\footnotesize \thepage\hfil\leftmark\hfil}   % The first page number is odd
        \def\@oddhead{\footnotesize \hfil\rightmark\hfil\thepage} } %
      { \def\@oddhead{\footnotesize \thepage\hfil\leftmark\hfil}    % The first page number is even
        \def\@evenhead{\footnotesize \hfil\rightmark\hfil\thepage} }%
      \let\@mkboth\markboth
}

\renewcommand\maketitle{\par 
   \begingroup
   \global\@topnum\z@
   \@maketitle
   \thispagestyle{firstpage} \@thanks
   \endgroup
   \global\let\thanks\relax
   \global\let\maketitle\relax
   \global\let\@maketitle\relax
   \global\let\@thanks\@empty
   \global\let\@author\@empty
   \global\let\@date\@empty
   \global\let\@title\@empty
   \global\let\title\relax
   \global\let\author\relax
   \global\let\date\relax
   \global\let\and\relax
}

%
\renewcommand\@maketitle{
  \normalfont\normalsize
  \global\topskip8\p@\relax % 5.5pc   "   "   "     "     "
  \vfil\null
%  \vskip 30\p@
  \@settitle
  \ifx\@empty\author 
     \else 
         \@setauthor
         \markboth 
          {\ifx\@empty\shortauthor {\footnotesize\@author}
\else {\footnotesize \noexpand\shortauthor} \fi
}
  \fi
  \ifx\@empty\@address \else \@setaddress \fi
%%%%%%%\rec
\noindent
  \@setabstract 
    %%%%%%%%%%\ifx\@empty\@ams\else \@setams\fi
  \vspace*{0.2mm}
  {\noindent \color{gray} \rule{\linewidth}{0.7mm}}
}

%%%%%%% \hrulefill
% footnote for corr author


%\setlength{\skip\footins}{2.5pc}
\renewcommand{\footnoterule}{\noindent\rule{5pc}{0.25pt}\vspace*{6pt}}
\renewcommand{\thefootnote}{\fnsymbol{footnote}}
\long\def\@makefntext#1{\parindent 1em
\noindent{$\m@th^{\@thefnmark}$}#1}

%\def\corraddr#1{\footnotetext[1]{Correspondence to: #1}\stepcounter{footnote}}
\def\corraddr#1{} % do not need address anymore!




%\def\corrauth{\footnotemark[1]\footnotetext[1]{Corresponding author.
\def\corrauth{\footnote{Corresponding author}\stepcounter{footnote}}
 %%%%%%\ifx\@empty\@email\else \@setemail\fi
 %%%%%%\ifx\@empty\@emails\else \@setemails\fi}\stepcounter{footnote}}



\def\affilnum#1{${}^{#1}$}
\def\affil#1{${}^{#1}$}
\def\comma{$^{\textrm{,}}$}

%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% Article titles
%
% Usage: \title[Short title]{Full title}
% [Short title] is optional; use where title is too long
% or contains footnotes, 50 characters maximum
%
\renewcommand*{\title}[2][]{\gdef\shorttitle{#1}\gdef\@title{#2}}
%\let\shorttitle=\empty
\def\@settitle{
\let \footnote \thanks
  \begin{center}
  \large  \bf
    \@title         
  \end{center}
  }

% List of authors
%
% Usage \author[Short form]{List of all authors}
% The short form excludes footnote symbols linking authors to addresses
% and is used for running heads in printed version (but not on preprints)
%
%\let\shortauthor=\empty
\renewcommand*{\author}[2][?]{
     \gdef\shortauthor{?} 
     \gdef\@author{#2}
   \ifthenelse{\equal{#1}{?}}
     { \gdef\shortauthor{\let\comma=\empty \let\corrauth=\empty \renewcommand{\affil}[1]{} #2} }
     { \gdef\shortauthor{#1}}
}
\def\@setauthor{\begin{center}
    \large \scshape
    \@author
    \end{center}%
}

%
% Affiliation (authors address)
%
% Usage: \address{Address of first author}
%        \address{Address of second author}
% Use once for each address, use symbols \dag \ddag \S \P $\|$
% to connect authors with addresses


\def\address#1{\def\@address{#1}}
\let\@address=\@empty
\def\@setaddress{\vspace{4pt}
	\begin{minipage}[b]{\dimexpr0.32\textwidth-1\fboxrule-0.5\fboxsep\relax}
		~~~~
	\end{minipage} \begin{minipage}[b]{0.68\textwidth}
   %\rm\small \@address
   {\rm\small \@address}
   \end{minipage}
}
%
\def\keywords#1{\def\@keywords{#1}}
\let\@keywords=\@empty
\def\@setkeywords{
	\addtolength{\topsep}{-4pt}
	\begin{trivlist}
		\item[]\itshape\small {\bf\small KEYWORDS} \newline \@keywords
	\end{trivlist}
}
%

%
%
%
% American Mathematical Society Classification Numbers
% Usage: \ams{57.XX, 58.XX}
%
\def\ams#1{\def\@ams{#1}}
\let\@ams=\@empty
\def\@setams{
     \addtolength{\topsep}{-4pt}
     \begin{trivlist}
     \item[]\rm\small {\bf \small AMS subject classifications}: \@ams
     \end{trivlist}
}
%
%
%
\newbox\abstractbox
\renewenvironment{abstract}{%
  \ifx\maketitle\relax
    \ClassWarning{\@classname}{Abstract should precede
      \protect\maketitle\space in AMS documentclasses; reported}%
  \fi
  \global\setbox\abstractbox=\vtop \bgroup
    \normalfont\small
    \list{}{\labelwidth\z@
           \leftmargin0pc %\rightmargin\leftmargin
       \itemindent\z@
      \parsep\z@ \@plus\p@
      \let\fullwidthdisplay\relax
    }
    \item[]
	 \begin{minipage}[b]{\dimexpr0.32\textwidth-1\fboxrule-0.5\fboxsep\relax}
	 		ARTICLE TYPE \newline
	 	\rm\small {\bf\type} \newline \newline \newline
	 	ARTICLE INFO \newline
	 Received: \datereceived \newline Accepted: \dateaccepted \newline DOI: \doinumber \newline JEL: \JEL  \newline \newline \newline
	\end{minipage}%
 		\begin{minipage}[b]{0.68\textwidth}
			\vspace{0.2cm} \item[\hskip\labelsep\bf SUMMARY]%
		\item[]%
	}{%
	\vspace{0.1cm}  \@setkeywords
	\end{minipage} \endlist\egroup}
\ifx\@setabstract\relax \@setabstracta \fi
\def\@setabstract{\@setabstracta \global\let\@setabstract\relax}
\def\@setabstracta{%
  \ifvoid\abstractbox
  \else
    \skip@6\p@ \advance\skip@-\lastskip
    \advance\skip@-\baselineskip \vskip\skip@
    \box\abstractbox
    \prevdepth\z@ % because \abstractbox is a vtop
  \fi
}


\def\proofBox{\hfill $\Box$}


%========================= Floats, tables, figures =================
%\renewcommand{\floatpagefraction}{1}
%\floatsep=10pt plus 2.5pt minus 1pt
%\textfloatsep=10pt plus 2.5pt minus 1pt
%\intextsep=10pt plus 2.5pt minus 1ptnewbox\abstractbox
%%\doublerulesep=0pt
%
%\abovecaptionskip=3pt
%\belowcaptionskip=5pt
%
%%
%% Insert \small and remove colon after table/figure number
%\long\def\@makecaption#1#2{%
%  \vskip\abovecaptionskip
%  \small
%  \sbox\@tempboxa{#1 #2}%
%  \ifdim \wd\@tempboxa >\hsize
%    #1 #2\par
%  \else
%    \global \@minipagefalse
%    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
%  \fi
%  \vskip\belowcaptionskip}
%
%\@addtoreset{table}{section}
%%\renewcommand\thetable{\thesection.\arabic{table}}
%%\def\fnum@table{\tablename\nobreakspace\thetable:}
%
%\@addtoreset{figure}{section}
%%\renewcommand\thefigure{\thesection.\arabic{figure}}

%\renewcommand\thefigure{\textsf \bf{figure}}
%\renewcommand\figurename{ \textsf{\bf Figure}}

%%\renewcommand\figurename{Fig.}
%%\def\fnum@figure{\figurename\nobreakspace\thefigure.}

\renewcommand{\floatpagefraction}{1}
\floatsep=10pt plus 2.5pt minus 1pt
\textfloatsep=10pt plus 2.5pt minus 1pt
\intextsep=10pt plus 2.5pt minus 1pt
%\doublerulesep=0pt

\abovecaptionskip=3pt
\belowcaptionskip=3pt

%%
%% Insert \small and remove colon after table/figure number
%\long\def\@makecaption#1#2{%
%  \vskip\abovecaptionskip
%  \small
%  \sbox\@tempboxa{#1 #2}%
%  \ifdim \wd\@tempboxa >\hsize
%    #1 #2\par
%  \else
%    \global \@minipagefalse
%    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
%  \fi
%  \vskip\belowcaptionskip}
  
% caption in small font
\long\def\@makecaption#1#2{%
  \renewcommand{\baselinestretch}{0.5}\rm
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\textsf{ #1.} \small \itshape{#2}}%
  \ifdim \wd\@tempboxa >\hsize
    {\sf \footnotesize #1. #2}\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip
  \renewcommand{\baselinestretch}{1}\rm
}  
  

%\@addtoreset{table}{section}
%\renewcommand\thetable{\thesection.\arabic{table}}
%\def\fnum@table{\tablename\nobreakspace\thetable:}
%
%\@addtoreset{figure}{section}
%\renewcommand\thefigure{\thesection.\arabic{figure}}
%\renewcommand\figurename{Fig.}
%\def\fnum@figure{\figurename\nobreakspace\thefigure.}



%======================== Section ===============================
\def\@seccntformat#1{\csname the#1\endcsname. }
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                     {-3.5ex \@plus -1ex \@minus -.2ex}%
                                     {2.3ex \@plus.2ex}%
                                     {\normalfont\large\bfseries}}
\renewcommand\subsection{\@startsection {subsection}{2}{\z@}%
                                         {-3.25ex\@plus -1ex \@minus -.2ex}%
                                        {1.5ex \@plus .2ex}%
                                        {\normalfont\large\bfseries}}

%======================== Bibliography ===============================
\renewcommand\@openbib@code{\parsep \z@ \itemsep \z@ \parskip \z@ \small}

%======================== Equation ===================================
%\@addtoreset{equation}{section}
%\renewcommand\theequation{\arabic{equation}}


\pagestyle{headings}

\renewcommand\markboth[2]{%
  \def\@evenhead{\thepage\hfill {\small #1}\hfill}%
  \def\@oddhead{\hfill{\small \itshape #2}\hfill \thepage}%
}



\markboth{}{}

\endinput

