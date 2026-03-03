import sys

content = r"""
\subsubsection{Preprocessing Agent}
Logika prapemrosesan (\textit{preprocessing}) melampaui pembersihan standar. Agen ini memanfaatkan pendekatan hibrida:
\begin{enumerate}
    \item \textbf{Normalisasi Teks}: Menghapus karakter yang tidak relevan, URL, dan menstandardisasi spasi putih menggunakan Regex.
    \item \textbf{Ekstraksi Fitur}: Menggunakan ekspresi reguler dan \textit{array} statistik (\texttt{numpy}) untuk mengekstraksi fitur seperti panjang teks, jumlah kata unik, keragaman leksikal, dan rasio huruf kapital.
\end{enumerate}

\subsubsection{Retriever Agent}
\textit{Retriever Agent} mengimplementasikan strategi pencarian semantik menggunakan indeks FAISS (\textit{Facebook AI Similarity Search}). Proses pengambilan ini dirumuskan dalam Algoritma \ref{alg:classifier}. Agen ini menghitung kemiripan kosinus (\textit{cosine similarity}) antara sematan (\textit{embedding}) artikel saat ini $\vec{v}_D$ dan sematan dalam basis pengetahuan $\vec{v}_K$. Jangkar (\textit{anchors}) top-K berfungsi sebagai titik pembelajaran dalam-konteks (\textit{in-context learning}/ICL) \textit{few-shot} untuk \textit{Classifier}.

\subsubsection{Classifier Agent}
Agen ini adalah "pengendali" dari proses penalaran. Agen ini dibangun di atas model Qwen 2.5 14B yang telah di-\textit{fine-tune}, dioptimalkan untuk nuansa jurnalistik Indonesia. Batas keputusan agen dibentuk oleh \textit{prompt} hierarkis yang mencakup lapisan metakognitif (mendefinisikan "persona"), lapisan domain (standar editorial berita), dan lapisan batasan (skema keluaran).

\subsubsection{Explanation Agent}
Agen ini beroperasi sebagai pasca-pemroses (\textit{post-processor}) untuk keluaran LLM. Agen ini memanfaatkan alat \texttt{SourceMapper} untuk menautkan klaim penalaran spesifik kembali ke cuplikan teks asli. Misalnya, jika \textit{Classifier} mengidentifikasi "Penyebutan merek yang berlebihan", \textit{Explanation Agent} memindai teks untuk memberikan frekuensi dan lokasi yang tepat dari nama merek tersebut, memberikan verifikasi empiris untuk klaim model asalnya.

\subsection{Kedalaman Algoritmik}
Logika operasional inti untuk penalaran berlandaskan RAG disajikan dalam Algoritma \ref{alg:classifier}, sementara strategi penyelarasan "Reasoning-First" yang khusus didefinisikan dalam Algoritma \ref{alg:prompt_alignment}.

\begin{algorithmic}
\STATE \textbf{Algoritma 1: Klasifikasi Agenik dengan RAG}
\REQUIRE Artikel Mentah $D$, Vektor KB $K$
\ENSURE Label Klasifikasi $L$, Jejak Penalaran $T$
\STATE $D^* \leftarrow S.Extract(D)$ \COMMENT{Ekstraksi akurasi tinggi}
\STATE $F \leftarrow P.distill(D^*)$ \COMMENT{Rekayasa fitur sintaksis}
\STATE $Context \leftarrow R.search(D^*, K, k=5)$ \COMMENT{Penjangkaran semantik}
\STATE $T, L \leftarrow C.reason(D^*, F, Context)$ \COMMENT{Eksekusi Reasoning-First}
\RETURN $L, T$
\end{algorithmic}

\begin{algorithmic}
\STATE \textbf{Algoritma 2: Penyelarasan Prompt Reasoning-First}
\REQUIRE Input $X$, Karakteristik $C_{markers}$, Contoh $E_{shots}$
\ENSURE Jejak Selaras $T$
\STATE Tentukan Identitas Sistem $I \leftarrow \text{"Senior News Editor"}$
\FOR{setiap $m \in C_{markers}$}
    \STATE $Step_m \leftarrow \text{Analisis } X \text{ untuk karakteristik } m$
\ENDFOR
\STATE $Context \leftarrow \text{Agregat } (Step_m)$
\STATE $T \leftarrow \text{SynthesizeContext}(I, Context, E_{shots})$
\RETURN $T$
\end{algorithmic}

\subsection{Rekayasa Prompt dan Penyelarasan Few-Shot}
Untuk memastikan akurasi klasifikasi yang tinggi (97\%) dan meminimalkan halusinasi, AGENA memanfaatkan \textit{prompt} sistem "Reasoning-First". Tidak seperti klasifikasi \textit{zero-shot} standar, \textit{Classifier Agent} diinstruksikan untuk melakukan analisis rantai-pemikiran (\textit{chain-of-thought}) sebelum mengeluarkan label akhir. Struktur \textit{prompt} didefinisikan sebagai berikut:

\begin{itemize}
    \item \textbf{Identitas Sistem}: Mendefinisikan agen sebagai editor senior dengan keahlian mendeteksi niat komersial yang tersembunyi.
    \item \textbf{Penjangkaran Karakteristik}: Memaksa agen untuk mengevaluasi teks terhadap empat penanda teoritis (Sentimen, Persuasi, Promosi, Perspektif).
    \item \textbf{Injeksi Kontekstual}: Menuntikkan top-5 artikel yang paling mirip yang diambil sebagai penanda batas semantik.
    \item \textbf{Batasan Respons}: Menerapkan skema JSON khusus untuk memfasilitasi penguraian terprogram dan pemetaan penjelasan di hilir.
\end{itemize}

Mesin penalaran inti (Qwen, Gemma, Llama, dan GPT-4o) yang digunakan dalam kerangka kerja AGENA direpresentasikan pada Gbr. \ref{fig:models}. Setiap model memiliki peran khusus, baik sebagai agen pengklasifikasi utama maupun sebagai hakim evaluasi.

Sebuah contoh dari templat \textit{few-shot} yang digunakan saat melatih (fine-tuning) model Qwen 2.5 14B disediakan dalam materi pelengkap, yang memastikan bahwa model mempelajari perbedaan bernuansa khusus antara "pelaporan positif" dan "bias promosi".

\subsection{Penyetelan Model dan Hyperparameter}
Mesin penalaran inti (Qwen, Gemma, Llama) dilatih ulang menggunakan teknik \textit{Parameter-Efficient Fine-Tuning} (PEFT), secara spesifik \textit{Low-Rank Adaptation} (LoRA) \cite{b24} dan kuantisasi 4-bit (QLoRA). Pendekatan ini memungkinkan optimalisasi model berskala besar di dalam batasan memori GPU yang tersedia dengan tetap mempertahankan performa \textit{state-of-the-art}. \textit{Hyperparameter} spesifik yang digunakan untuk urutan v6, yang mencapai akurasi 97\%, dirinci dalam Tabel \ref{tab:hyperparameters}.

\begin{table}[htbp]
\caption{Hyperparameter Fine-tuning di Berbagai Model}
\label{tab:hyperparameters}
\begin{center}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Hyperparameter} & \textbf{Qwen 14B} & \textbf{Gemma 9B} & \textbf{Llama 8B} \\
\hline
Peringkat LoRA (\textit{Rank} $r$) & 32 & 16 & 16 \\
LoRA Alpha ($\alpha$) & 64 & 32 & 32 \\
Laju Pembelajaran (\textit{Learning Rate}) & 2e-5 & 5e-5 & 3e-5 \\
Ukuran Batch (\textit{Batch Size}) & 16 & 8 & 8 \\
Epos (\textit{Epochs}) & 3 & 5 & 5 \\
Kuantisasi & 4-bit & 4-bit & 4-bit \\
Pengoptimal (\textit{Optimizer}) & AdamW & AdamW & AdamW \\
\hline
\end{tabular}
\end{center}
\end{table}
"""

with open('/Users/brianrizqi/Documents/Post Doc/agentic-ai-native-ads/langchain-refactor/article/example.tex', 'a') as f:
    f.write(content)

