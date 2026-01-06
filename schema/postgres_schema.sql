--
-- PostgreSQL database dump
--


-- Dumped from database version 16.11 (Ubuntu 16.11-0ubuntu0.24.04.1)
-- Dumped by pg_dump version 16.11 (Ubuntu 16.11-0ubuntu0.24.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: pg_trgm; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;


--
-- Name: EXTENSION pg_trgm; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pg_trgm IS 'text similarity measurement and index searching based on trigrams';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: sync_concept_domains(); Type: FUNCTION; Schema: public; Owner: polymath
--

CREATE FUNCTION public.sync_concept_domains() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.concept_domains := ARRAY(
        SELECT jsonb_array_elements_text(
            COALESCE(NEW.meta_json->'concept_domains', '[]'::jsonb)
        )
    );
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.sync_concept_domains() OWNER TO polymath;

--
-- Name: update_artifact_tsv(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_artifact_tsv() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', 
        COALESCE(NEW.title, '') || ' ' || 
        COALESCE(array_to_string(NEW.authors, ' '), ''));
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_artifact_tsv() OWNER TO postgres;

--
-- Name: update_chunk_tsv(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_chunk_tsv() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_chunk_tsv() OWNER TO postgres;

--
-- Name: update_code_search_vector(); Type: FUNCTION; Schema: public; Owner: polymath
--

CREATE FUNCTION public.update_code_search_vector() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.search_vector := 
        setweight(to_tsvector('english', COALESCE(NEW.name, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.signature, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.docstring, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'C');
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_code_search_vector() OWNER TO polymath;

--
-- Name: update_updated_at(); Type: FUNCTION; Schema: public; Owner: polymath
--

CREATE FUNCTION public.update_updated_at() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_updated_at() OWNER TO polymath;

--
-- Name: update_updated_at_column(); Type: FUNCTION; Schema: public; Owner: polymath
--

CREATE FUNCTION public.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_updated_at_column() OWNER TO polymath;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: artifact_concepts; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.artifact_concepts (
    artifact_id uuid NOT NULL,
    concept_id integer NOT NULL,
    confidence double precision DEFAULT 1.0
);


ALTER TABLE public.artifact_concepts OWNER TO postgres;

--
-- Name: artifacts; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.artifacts (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    artifact_type text NOT NULL,
    title text NOT NULL,
    file_path text,
    file_hash text,
    source_url text,
    repo text,
    commit_hash text,
    authors text[],
    year integer,
    license text,
    language text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    indexed_at timestamp with time zone,
    content_tsv tsvector,
    source_item_id integer,
    ingest_run_id uuid,
    doc_id uuid,
    verified_doi boolean DEFAULT false,
    publication_type text,
    CONSTRAINT artifacts_artifact_type_check CHECK ((artifact_type = ANY (ARRAY['paper'::text, 'code_file'::text, 'code_repo'::text, 'config'::text, 'notebook'::text, 'log'::text, 'textbook'::text])))
);


ALTER TABLE public.artifacts OWNER TO postgres;

--
-- Name: benchmark_runs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.benchmark_runs (
    id integer NOT NULL,
    benchmark_id integer,
    model_name text NOT NULL,
    config jsonb,
    metric_value double precision NOT NULL,
    run_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.benchmark_runs OWNER TO postgres;

--
-- Name: benchmark_runs_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.benchmark_runs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.benchmark_runs_id_seq OWNER TO postgres;

--
-- Name: benchmark_runs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.benchmark_runs_id_seq OWNED BY public.benchmark_runs.id;


--
-- Name: benchmarks; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.benchmarks (
    id integer NOT NULL,
    name text NOT NULL,
    metric_name text NOT NULL,
    higher_is_better boolean DEFAULT true
);


ALTER TABLE public.benchmarks OWNER TO postgres;

--
-- Name: benchmarks_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.benchmarks_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.benchmarks_id_seq OWNER TO postgres;

--
-- Name: benchmarks_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.benchmarks_id_seq OWNED BY public.benchmarks.id;


--
-- Name: chunks; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.chunks (
    id text NOT NULL,
    artifact_id uuid,
    chunk_index integer NOT NULL,
    start_line integer,
    end_line integer,
    content text NOT NULL,
    content_tsv tsvector,
    created_at timestamp with time zone DEFAULT now(),
    is_citation_eligible boolean DEFAULT false
);


ALTER TABLE public.chunks OWNER TO postgres;

--
-- Name: claim_evidence_links; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.claim_evidence_links (
    link_id integer NOT NULL,
    response_id uuid,
    claim_id text NOT NULL,
    claim_text text NOT NULL,
    span_id integer NOT NULL,
    entailment_score double precision NOT NULL,
    support_type text,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.claim_evidence_links OWNER TO postgres;

--
-- Name: claim_evidence_links_link_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.claim_evidence_links_link_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.claim_evidence_links_link_id_seq OWNER TO postgres;

--
-- Name: claim_evidence_links_link_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.claim_evidence_links_link_id_seq OWNED BY public.claim_evidence_links.link_id;


--
-- Name: code_chunks; Type: TABLE; Schema: public; Owner: polymath
--

CREATE TABLE public.code_chunks (
    chunk_id uuid DEFAULT gen_random_uuid() NOT NULL,
    file_id uuid,
    chunk_type text,
    name text,
    class_name text,
    symbol_qualified_name text,
    start_line integer,
    end_line integer,
    content text NOT NULL,
    chunk_hash text,
    docstring text,
    signature text,
    imports text[],
    concepts text[],
    search_vector tsvector,
    embedding_id text,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.code_chunks OWNER TO polymath;

--
-- Name: code_files; Type: TABLE; Schema: public; Owner: polymath
--

CREATE TABLE public.code_files (
    file_id uuid DEFAULT gen_random_uuid() NOT NULL,
    repo_name text NOT NULL,
    repo_url text,
    repo_root text,
    default_branch text DEFAULT 'main'::text,
    head_commit_sha text,
    file_path text NOT NULL,
    language text,
    file_hash text,
    file_size_bytes integer,
    loc integer,
    is_generated boolean DEFAULT false,
    modified_time timestamp with time zone,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.code_files OWNER TO polymath;

--
-- Name: code_references; Type: TABLE; Schema: public; Owner: polymath
--

CREATE TABLE public.code_references (
    id integer NOT NULL,
    chunk_id uuid,
    ref_type text NOT NULL,
    ref_doc_id uuid,
    ref_concept_id integer,
    ref_chunk_id uuid,
    confidence double precision DEFAULT 1.0,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT exactly_one_ref CHECK ((((((ref_doc_id IS NOT NULL))::integer + ((ref_concept_id IS NOT NULL))::integer) + ((ref_chunk_id IS NOT NULL))::integer) = 1))
);


ALTER TABLE public.code_references OWNER TO polymath;

--
-- Name: code_references_id_seq; Type: SEQUENCE; Schema: public; Owner: polymath
--

CREATE SEQUENCE public.code_references_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.code_references_id_seq OWNER TO polymath;

--
-- Name: code_references_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: polymath
--

ALTER SEQUENCE public.code_references_id_seq OWNED BY public.code_references.id;


--
-- Name: concepts; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.concepts (
    id integer NOT NULL,
    name text NOT NULL,
    domain text,
    is_cross_domain boolean DEFAULT false
);


ALTER TABLE public.concepts OWNER TO postgres;

--
-- Name: concepts_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.concepts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.concepts_id_seq OWNER TO postgres;

--
-- Name: concepts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.concepts_id_seq OWNED BY public.concepts.id;


--
-- Name: doc_aliases; Type: TABLE; Schema: public; Owner: polymath
--

CREATE TABLE public.doc_aliases (
    alias_id integer NOT NULL,
    doc_id uuid NOT NULL,
    alias_type text NOT NULL,
    alias_value text NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.doc_aliases OWNER TO polymath;

--
-- Name: doc_aliases_alias_id_seq; Type: SEQUENCE; Schema: public; Owner: polymath
--

CREATE SEQUENCE public.doc_aliases_alias_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.doc_aliases_alias_id_seq OWNER TO polymath;

--
-- Name: doc_aliases_alias_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: polymath
--

ALTER SEQUENCE public.doc_aliases_alias_id_seq OWNED BY public.doc_aliases.alias_id;


--
-- Name: documents; Type: TABLE; Schema: public; Owner: polymath
--

CREATE TABLE public.documents (
    doc_id uuid NOT NULL,
    doi text,
    pmid text,
    arxiv_id text,
    title_hash text,
    title text NOT NULL,
    authors text[],
    year integer,
    venue text,
    publication_type text,
    parser_version text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.documents OWNER TO polymath;

--
-- Name: download_jobs; Type: TABLE; Schema: public; Owner: polymath
--

CREATE TABLE public.download_jobs (
    id integer NOT NULL,
    source_item_id integer,
    attempt integer DEFAULT 1,
    status text DEFAULT 'pending'::text,
    file_path text,
    file_size bigint,
    sha256 text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.download_jobs OWNER TO polymath;

--
-- Name: download_jobs_id_seq; Type: SEQUENCE; Schema: public; Owner: polymath
--

CREATE SEQUENCE public.download_jobs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.download_jobs_id_seq OWNER TO polymath;

--
-- Name: download_jobs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: polymath
--

ALTER SEQUENCE public.download_jobs_id_seq OWNED BY public.download_jobs.id;


--
-- Name: eval_results; Type: TABLE; Schema: public; Owner: polymath
--

CREATE TABLE public.eval_results (
    id integer NOT NULL,
    run_id uuid,
    eval_suite text,
    score double precision,
    details_json jsonb,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.eval_results OWNER TO polymath;

--
-- Name: eval_results_id_seq; Type: SEQUENCE; Schema: public; Owner: polymath
--

CREATE SEQUENCE public.eval_results_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.eval_results_id_seq OWNER TO polymath;

--
-- Name: eval_results_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: polymath
--

ALTER SEQUENCE public.eval_results_id_seq OWNED BY public.eval_results.id;


--
-- Name: evidence_spans; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.evidence_spans (
    span_id integer NOT NULL,
    passage_id uuid NOT NULL,
    doc_id uuid NOT NULL,
    page_num integer NOT NULL,
    sentence_idx integer,
    span_char_start integer NOT NULL,
    span_char_end integer NOT NULL,
    span_text text NOT NULL,
    section text,
    quality_score double precision,
    entailment_score double precision,
    contradiction_score double precision,
    created_at timestamp with time zone DEFAULT now(),
    source_type text DEFAULT 'passage'::text,
    chunk_id text
);


ALTER TABLE public.evidence_spans OWNER TO postgres;

--
-- Name: evidence_spans_span_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.evidence_spans_span_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.evidence_spans_span_id_seq OWNER TO postgres;

--
-- Name: evidence_spans_span_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.evidence_spans_span_id_seq OWNED BY public.evidence_spans.span_id;


--
-- Name: failure_cards; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.failure_cards (
    id integer NOT NULL,
    description text NOT NULL,
    root_cause text,
    fix_applied text,
    severity text,
    resolved boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.failure_cards OWNER TO postgres;

--
-- Name: failure_cards_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.failure_cards_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.failure_cards_id_seq OWNER TO postgres;

--
-- Name: failure_cards_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.failure_cards_id_seq OWNED BY public.failure_cards.id;


--
-- Name: ingest_runs; Type: TABLE; Schema: public; Owner: polymath
--

CREATE TABLE public.ingest_runs (
    run_id uuid DEFAULT gen_random_uuid() NOT NULL,
    started_at timestamp with time zone DEFAULT now(),
    finished_at timestamp with time zone,
    status text DEFAULT 'started'::text,
    source_items_processed integer DEFAULT 0,
    chunks_added integer DEFAULT 0,
    concepts_linked integer DEFAULT 0,
    git_sha text,
    counts_json jsonb,
    error_log text
);


ALTER TABLE public.ingest_runs OWNER TO polymath;

--
-- Name: needs_user_tickets; Type: TABLE; Schema: public; Owner: polymath
--

CREATE TABLE public.needs_user_tickets (
    id integer NOT NULL,
    source_item_id integer,
    doi text,
    title text,
    url text,
    reason text,
    resolved boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.needs_user_tickets OWNER TO polymath;

--
-- Name: needs_user_tickets_id_seq; Type: SEQUENCE; Schema: public; Owner: polymath
--

CREATE SEQUENCE public.needs_user_tickets_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.needs_user_tickets_id_seq OWNER TO polymath;

--
-- Name: needs_user_tickets_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: polymath
--

ALTER SEQUENCE public.needs_user_tickets_id_seq OWNED BY public.needs_user_tickets.id;


--
-- Name: passages; Type: TABLE; Schema: public; Owner: polymath
--

CREATE TABLE public.passages (
    passage_id uuid NOT NULL,
    doc_id uuid NOT NULL,
    page_num integer,
    page_char_start integer NOT NULL,
    page_char_end integer NOT NULL,
    section text,
    passage_text text NOT NULL,
    quality_score double precision DEFAULT 1.0,
    parser_version text,
    created_at timestamp with time zone DEFAULT now(),
    citable boolean DEFAULT true,
    CONSTRAINT passages_page_num_check CHECK ((page_num >= '-1'::integer))
);


ALTER TABLE public.passages OWNER TO polymath;

--
-- Name: recipe_cards; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.recipe_cards (
    id integer NOT NULL,
    title text NOT NULL,
    problem text NOT NULL,
    solution text NOT NULL,
    artifacts uuid[],
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.recipe_cards OWNER TO postgres;

--
-- Name: recipe_cards_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.recipe_cards_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.recipe_cards_id_seq OWNER TO postgres;

--
-- Name: recipe_cards_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.recipe_cards_id_seq OWNED BY public.recipe_cards.id;


--
-- Name: source_items; Type: TABLE; Schema: public; Owner: polymath
--

CREATE TABLE public.source_items (
    id integer NOT NULL,
    source text NOT NULL,
    external_id text NOT NULL,
    url text,
    title text,
    authors text[],
    published_at timestamp with time zone,
    doi text,
    sha256 text,
    status text DEFAULT 'discovered'::text,
    last_error text,
    meta_json jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    priority_score double precision GENERATED ALWAYS AS (COALESCE(((meta_json ->> 'priority_score'::text))::double precision, (0.0)::double precision)) STORED,
    concept_domains text[] DEFAULT '{}'::text[],
    is_hidden_gem boolean GENERATED ALWAYS AS (COALESCE(((meta_json ->> 'is_hidden_gem'::text))::boolean, false)) STORED
);


ALTER TABLE public.source_items OWNER TO polymath;

--
-- Name: source_items_id_seq; Type: SEQUENCE; Schema: public; Owner: polymath
--

CREATE SEQUENCE public.source_items_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.source_items_id_seq OWNER TO polymath;

--
-- Name: source_items_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: polymath
--

ALTER SEQUENCE public.source_items_id_seq OWNED BY public.source_items.id;


--
-- Name: benchmark_runs id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.benchmark_runs ALTER COLUMN id SET DEFAULT nextval('public.benchmark_runs_id_seq'::regclass);


--
-- Name: benchmarks id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.benchmarks ALTER COLUMN id SET DEFAULT nextval('public.benchmarks_id_seq'::regclass);


--
-- Name: claim_evidence_links link_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.claim_evidence_links ALTER COLUMN link_id SET DEFAULT nextval('public.claim_evidence_links_link_id_seq'::regclass);


--
-- Name: code_references id; Type: DEFAULT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.code_references ALTER COLUMN id SET DEFAULT nextval('public.code_references_id_seq'::regclass);


--
-- Name: concepts id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.concepts ALTER COLUMN id SET DEFAULT nextval('public.concepts_id_seq'::regclass);


--
-- Name: doc_aliases alias_id; Type: DEFAULT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.doc_aliases ALTER COLUMN alias_id SET DEFAULT nextval('public.doc_aliases_alias_id_seq'::regclass);


--
-- Name: download_jobs id; Type: DEFAULT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.download_jobs ALTER COLUMN id SET DEFAULT nextval('public.download_jobs_id_seq'::regclass);


--
-- Name: eval_results id; Type: DEFAULT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.eval_results ALTER COLUMN id SET DEFAULT nextval('public.eval_results_id_seq'::regclass);


--
-- Name: evidence_spans span_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.evidence_spans ALTER COLUMN span_id SET DEFAULT nextval('public.evidence_spans_span_id_seq'::regclass);


--
-- Name: failure_cards id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.failure_cards ALTER COLUMN id SET DEFAULT nextval('public.failure_cards_id_seq'::regclass);


--
-- Name: needs_user_tickets id; Type: DEFAULT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.needs_user_tickets ALTER COLUMN id SET DEFAULT nextval('public.needs_user_tickets_id_seq'::regclass);


--
-- Name: recipe_cards id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.recipe_cards ALTER COLUMN id SET DEFAULT nextval('public.recipe_cards_id_seq'::regclass);


--
-- Name: source_items id; Type: DEFAULT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.source_items ALTER COLUMN id SET DEFAULT nextval('public.source_items_id_seq'::regclass);


--
-- Name: artifact_concepts artifact_concepts_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_concepts
    ADD CONSTRAINT artifact_concepts_pkey PRIMARY KEY (artifact_id, concept_id);


--
-- Name: artifacts artifacts_file_hash_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifacts
    ADD CONSTRAINT artifacts_file_hash_key UNIQUE (file_hash);


--
-- Name: artifacts artifacts_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifacts
    ADD CONSTRAINT artifacts_pkey PRIMARY KEY (id);


--
-- Name: benchmark_runs benchmark_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.benchmark_runs
    ADD CONSTRAINT benchmark_runs_pkey PRIMARY KEY (id);


--
-- Name: benchmarks benchmarks_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.benchmarks
    ADD CONSTRAINT benchmarks_pkey PRIMARY KEY (id);


--
-- Name: chunks chunks_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chunks
    ADD CONSTRAINT chunks_pkey PRIMARY KEY (id);


--
-- Name: claim_evidence_links claim_evidence_links_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.claim_evidence_links
    ADD CONSTRAINT claim_evidence_links_pkey PRIMARY KEY (link_id);


--
-- Name: code_chunks code_chunks_pkey; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.code_chunks
    ADD CONSTRAINT code_chunks_pkey PRIMARY KEY (chunk_id);


--
-- Name: code_files code_files_pkey; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.code_files
    ADD CONSTRAINT code_files_pkey PRIMARY KEY (file_id);


--
-- Name: code_files code_files_repo_name_file_path_head_commit_sha_key; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.code_files
    ADD CONSTRAINT code_files_repo_name_file_path_head_commit_sha_key UNIQUE (repo_name, file_path, head_commit_sha);


--
-- Name: code_references code_references_pkey; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.code_references
    ADD CONSTRAINT code_references_pkey PRIMARY KEY (id);


--
-- Name: concepts concepts_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.concepts
    ADD CONSTRAINT concepts_name_key UNIQUE (name);


--
-- Name: concepts concepts_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.concepts
    ADD CONSTRAINT concepts_pkey PRIMARY KEY (id);


--
-- Name: doc_aliases doc_aliases_alias_type_alias_value_key; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.doc_aliases
    ADD CONSTRAINT doc_aliases_alias_type_alias_value_key UNIQUE (alias_type, alias_value);


--
-- Name: doc_aliases doc_aliases_pkey; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.doc_aliases
    ADD CONSTRAINT doc_aliases_pkey PRIMARY KEY (alias_id);


--
-- Name: documents documents_arxiv_id_key; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_arxiv_id_key UNIQUE (arxiv_id);


--
-- Name: documents documents_doi_key; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_doi_key UNIQUE (doi);


--
-- Name: documents documents_pkey; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_pkey PRIMARY KEY (doc_id);


--
-- Name: documents documents_pmid_key; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_pmid_key UNIQUE (pmid);


--
-- Name: documents documents_title_hash_key; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_title_hash_key UNIQUE (title_hash);


--
-- Name: download_jobs download_jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.download_jobs
    ADD CONSTRAINT download_jobs_pkey PRIMARY KEY (id);


--
-- Name: eval_results eval_results_pkey; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.eval_results
    ADD CONSTRAINT eval_results_pkey PRIMARY KEY (id);


--
-- Name: evidence_spans evidence_spans_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.evidence_spans
    ADD CONSTRAINT evidence_spans_pkey PRIMARY KEY (span_id);


--
-- Name: failure_cards failure_cards_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.failure_cards
    ADD CONSTRAINT failure_cards_pkey PRIMARY KEY (id);


--
-- Name: ingest_runs ingest_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.ingest_runs
    ADD CONSTRAINT ingest_runs_pkey PRIMARY KEY (run_id);


--
-- Name: needs_user_tickets needs_user_tickets_pkey; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.needs_user_tickets
    ADD CONSTRAINT needs_user_tickets_pkey PRIMARY KEY (id);


--
-- Name: passages passages_pkey; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.passages
    ADD CONSTRAINT passages_pkey PRIMARY KEY (passage_id);


--
-- Name: recipe_cards recipe_cards_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.recipe_cards
    ADD CONSTRAINT recipe_cards_pkey PRIMARY KEY (id);


--
-- Name: source_items source_items_pkey; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.source_items
    ADD CONSTRAINT source_items_pkey PRIMARY KEY (id);


--
-- Name: source_items source_items_source_external_id_key; Type: CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.source_items
    ADD CONSTRAINT source_items_source_external_id_key UNIQUE (source, external_id);


--
-- Name: idx_artifacts_content_tsv; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_artifacts_content_tsv ON public.artifacts USING gin (content_tsv);


--
-- Name: idx_artifacts_doc; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_artifacts_doc ON public.artifacts USING btree (doc_id);


--
-- Name: idx_artifacts_title_trgm; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_artifacts_title_trgm ON public.artifacts USING gin (title public.gin_trgm_ops);


--
-- Name: idx_artifacts_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_artifacts_type ON public.artifacts USING btree (artifact_type);


--
-- Name: idx_chunks_artifact; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_artifact ON public.chunks USING btree (artifact_id);


--
-- Name: idx_chunks_content_tsv; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_content_tsv ON public.chunks USING gin (content_tsv);


--
-- Name: idx_code_chunks_class; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_code_chunks_class ON public.code_chunks USING btree (class_name);


--
-- Name: idx_code_chunks_file_id; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_code_chunks_file_id ON public.code_chunks USING btree (file_id);


--
-- Name: idx_code_chunks_fts; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_code_chunks_fts ON public.code_chunks USING gin (search_vector);


--
-- Name: idx_code_chunks_name; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_code_chunks_name ON public.code_chunks USING btree (name);


--
-- Name: idx_code_chunks_symbol; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_code_chunks_symbol ON public.code_chunks USING btree (symbol_qualified_name);


--
-- Name: idx_code_chunks_type; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_code_chunks_type ON public.code_chunks USING btree (chunk_type);


--
-- Name: idx_code_files_commit; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_code_files_commit ON public.code_files USING btree (head_commit_sha);


--
-- Name: idx_code_files_lang; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_code_files_lang ON public.code_files USING btree (language);


--
-- Name: idx_code_files_repo; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_code_files_repo ON public.code_files USING btree (repo_name);


--
-- Name: idx_code_refs_chunk; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_code_refs_chunk ON public.code_references USING btree (chunk_id);


--
-- Name: idx_doc_aliases_doc; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_doc_aliases_doc ON public.doc_aliases USING btree (doc_id);


--
-- Name: idx_doc_aliases_lookup; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_doc_aliases_lookup ON public.doc_aliases USING btree (alias_type, alias_value);


--
-- Name: idx_documents_arxiv; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_documents_arxiv ON public.documents USING btree (arxiv_id) WHERE (arxiv_id IS NOT NULL);


--
-- Name: idx_documents_doi; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_documents_doi ON public.documents USING btree (doi) WHERE (doi IS NOT NULL);


--
-- Name: idx_documents_pmid; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_documents_pmid ON public.documents USING btree (pmid) WHERE (pmid IS NOT NULL);


--
-- Name: idx_documents_title_hash; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_documents_title_hash ON public.documents USING btree (title_hash);


--
-- Name: idx_documents_year; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_documents_year ON public.documents USING btree (year) WHERE (year IS NOT NULL);


--
-- Name: idx_download_jobs_source_item; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_download_jobs_source_item ON public.download_jobs USING btree (source_item_id);


--
-- Name: idx_download_jobs_status; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_download_jobs_status ON public.download_jobs USING btree (status);


--
-- Name: idx_evidence_spans_source_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_evidence_spans_source_type ON public.evidence_spans USING btree (source_type);


--
-- Name: idx_ingest_runs_started; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_ingest_runs_started ON public.ingest_runs USING btree (started_at DESC);


--
-- Name: idx_ingest_runs_status; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_ingest_runs_status ON public.ingest_runs USING btree (status);


--
-- Name: idx_links_claim; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_links_claim ON public.claim_evidence_links USING btree (claim_id);


--
-- Name: idx_links_response; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_links_response ON public.claim_evidence_links USING btree (response_id);


--
-- Name: idx_links_span; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_links_span ON public.claim_evidence_links USING btree (span_id);


--
-- Name: idx_links_support; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_links_support ON public.claim_evidence_links USING btree (support_type);


--
-- Name: idx_needs_user_tickets_created; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_needs_user_tickets_created ON public.needs_user_tickets USING btree (created_at DESC);


--
-- Name: idx_needs_user_tickets_resolved; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_needs_user_tickets_resolved ON public.needs_user_tickets USING btree (resolved);


--
-- Name: idx_passages_doc; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_passages_doc ON public.passages USING btree (doc_id);


--
-- Name: idx_passages_page; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_passages_page ON public.passages USING btree (doc_id, page_num);


--
-- Name: idx_passages_quality; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_passages_quality ON public.passages USING btree (quality_score) WHERE (quality_score >= (0.7)::double precision);


--
-- Name: idx_passages_section; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_passages_section ON public.passages USING btree (section);


--
-- Name: idx_source_items_concepts; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_source_items_concepts ON public.source_items USING gin (concept_domains);


--
-- Name: idx_source_items_created; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_source_items_created ON public.source_items USING btree (created_at DESC);


--
-- Name: idx_source_items_doi; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_source_items_doi ON public.source_items USING btree (doi);


--
-- Name: idx_source_items_hidden_gem; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_source_items_hidden_gem ON public.source_items USING btree (is_hidden_gem) WHERE (is_hidden_gem = true);


--
-- Name: idx_source_items_priority; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_source_items_priority ON public.source_items USING btree (priority_score DESC NULLS LAST);


--
-- Name: idx_source_items_source; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_source_items_source ON public.source_items USING btree (source);


--
-- Name: idx_source_items_status; Type: INDEX; Schema: public; Owner: polymath
--

CREATE INDEX idx_source_items_status ON public.source_items USING btree (status);


--
-- Name: idx_spans_doc; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spans_doc ON public.evidence_spans USING btree (doc_id);


--
-- Name: idx_spans_entailment; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spans_entailment ON public.evidence_spans USING btree (entailment_score) WHERE (entailment_score > (0.5)::double precision);


--
-- Name: idx_spans_passage; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_spans_passage ON public.evidence_spans USING btree (passage_id);


--
-- Name: idx_spans_unique; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX idx_spans_unique ON public.evidence_spans USING btree (passage_id, sentence_idx);


--
-- Name: artifacts artifact_tsv_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER artifact_tsv_update BEFORE INSERT OR UPDATE ON public.artifacts FOR EACH ROW EXECUTE FUNCTION public.update_artifact_tsv();


--
-- Name: chunks chunk_tsv_update; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER chunk_tsv_update BEFORE INSERT OR UPDATE ON public.chunks FOR EACH ROW EXECUTE FUNCTION public.update_chunk_tsv();


--
-- Name: code_chunks code_chunks_search_update; Type: TRIGGER; Schema: public; Owner: polymath
--

CREATE TRIGGER code_chunks_search_update BEFORE INSERT OR UPDATE ON public.code_chunks FOR EACH ROW EXECUTE FUNCTION public.update_code_search_vector();


--
-- Name: download_jobs download_jobs_updated_at; Type: TRIGGER; Schema: public; Owner: polymath
--

CREATE TRIGGER download_jobs_updated_at BEFORE UPDATE ON public.download_jobs FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();


--
-- Name: source_items source_items_updated_at; Type: TRIGGER; Schema: public; Owner: polymath
--

CREATE TRIGGER source_items_updated_at BEFORE UPDATE ON public.source_items FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();


--
-- Name: source_items sync_concept_domains_trigger; Type: TRIGGER; Schema: public; Owner: polymath
--

CREATE TRIGGER sync_concept_domains_trigger BEFORE INSERT OR UPDATE ON public.source_items FOR EACH ROW EXECUTE FUNCTION public.sync_concept_domains();


--
-- Name: documents update_documents_updated_at; Type: TRIGGER; Schema: public; Owner: polymath
--

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON public.documents FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: artifact_concepts artifact_concepts_artifact_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_concepts
    ADD CONSTRAINT artifact_concepts_artifact_id_fkey FOREIGN KEY (artifact_id) REFERENCES public.artifacts(id) ON DELETE CASCADE;


--
-- Name: artifact_concepts artifact_concepts_concept_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_concepts
    ADD CONSTRAINT artifact_concepts_concept_id_fkey FOREIGN KEY (concept_id) REFERENCES public.concepts(id) ON DELETE CASCADE;


--
-- Name: artifacts artifacts_doc_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifacts
    ADD CONSTRAINT artifacts_doc_id_fkey FOREIGN KEY (doc_id) REFERENCES public.documents(doc_id) ON DELETE SET NULL;


--
-- Name: artifacts artifacts_ingest_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifacts
    ADD CONSTRAINT artifacts_ingest_run_id_fkey FOREIGN KEY (ingest_run_id) REFERENCES public.ingest_runs(run_id);


--
-- Name: artifacts artifacts_source_item_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifacts
    ADD CONSTRAINT artifacts_source_item_id_fkey FOREIGN KEY (source_item_id) REFERENCES public.source_items(id);


--
-- Name: benchmark_runs benchmark_runs_benchmark_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.benchmark_runs
    ADD CONSTRAINT benchmark_runs_benchmark_id_fkey FOREIGN KEY (benchmark_id) REFERENCES public.benchmarks(id);


--
-- Name: chunks chunks_artifact_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chunks
    ADD CONSTRAINT chunks_artifact_id_fkey FOREIGN KEY (artifact_id) REFERENCES public.artifacts(id) ON DELETE CASCADE;


--
-- Name: claim_evidence_links claim_evidence_links_span_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.claim_evidence_links
    ADD CONSTRAINT claim_evidence_links_span_id_fkey FOREIGN KEY (span_id) REFERENCES public.evidence_spans(span_id) ON DELETE CASCADE;


--
-- Name: code_chunks code_chunks_file_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.code_chunks
    ADD CONSTRAINT code_chunks_file_id_fkey FOREIGN KEY (file_id) REFERENCES public.code_files(file_id) ON DELETE CASCADE;


--
-- Name: code_references code_references_chunk_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.code_references
    ADD CONSTRAINT code_references_chunk_id_fkey FOREIGN KEY (chunk_id) REFERENCES public.code_chunks(chunk_id) ON DELETE CASCADE;


--
-- Name: code_references code_references_ref_chunk_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.code_references
    ADD CONSTRAINT code_references_ref_chunk_id_fkey FOREIGN KEY (ref_chunk_id) REFERENCES public.code_chunks(chunk_id) ON DELETE SET NULL;


--
-- Name: code_references code_references_ref_doc_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.code_references
    ADD CONSTRAINT code_references_ref_doc_id_fkey FOREIGN KEY (ref_doc_id) REFERENCES public.documents(doc_id) ON DELETE SET NULL;


--
-- Name: doc_aliases doc_aliases_doc_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.doc_aliases
    ADD CONSTRAINT doc_aliases_doc_id_fkey FOREIGN KEY (doc_id) REFERENCES public.documents(doc_id) ON DELETE CASCADE;


--
-- Name: download_jobs download_jobs_source_item_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.download_jobs
    ADD CONSTRAINT download_jobs_source_item_id_fkey FOREIGN KEY (source_item_id) REFERENCES public.source_items(id);


--
-- Name: eval_results eval_results_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.eval_results
    ADD CONSTRAINT eval_results_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.ingest_runs(run_id);


--
-- Name: evidence_spans evidence_spans_doc_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.evidence_spans
    ADD CONSTRAINT evidence_spans_doc_id_fkey FOREIGN KEY (doc_id) REFERENCES public.documents(doc_id) ON DELETE CASCADE;


--
-- Name: evidence_spans evidence_spans_passage_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.evidence_spans
    ADD CONSTRAINT evidence_spans_passage_id_fkey FOREIGN KEY (passage_id) REFERENCES public.passages(passage_id) ON DELETE CASCADE;


--
-- Name: needs_user_tickets needs_user_tickets_source_item_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.needs_user_tickets
    ADD CONSTRAINT needs_user_tickets_source_item_id_fkey FOREIGN KEY (source_item_id) REFERENCES public.source_items(id);


--
-- Name: passages passages_doc_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: polymath
--

ALTER TABLE ONLY public.passages
    ADD CONSTRAINT passages_doc_id_fkey FOREIGN KEY (doc_id) REFERENCES public.documents(doc_id) ON DELETE CASCADE;


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: pg_database_owner
--

GRANT ALL ON SCHEMA public TO polymath;


--
-- Name: TABLE artifact_concepts; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.artifact_concepts TO polymath;


--
-- Name: TABLE artifacts; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.artifacts TO polymath;


--
-- Name: TABLE benchmark_runs; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.benchmark_runs TO polymath;


--
-- Name: SEQUENCE benchmark_runs_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.benchmark_runs_id_seq TO polymath;


--
-- Name: TABLE benchmarks; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.benchmarks TO polymath;


--
-- Name: SEQUENCE benchmarks_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.benchmarks_id_seq TO polymath;


--
-- Name: TABLE chunks; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.chunks TO polymath;


--
-- Name: TABLE claim_evidence_links; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.claim_evidence_links TO polymath;


--
-- Name: SEQUENCE claim_evidence_links_link_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.claim_evidence_links_link_id_seq TO polymath;


--
-- Name: TABLE concepts; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.concepts TO polymath;


--
-- Name: SEQUENCE concepts_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.concepts_id_seq TO polymath;


--
-- Name: TABLE evidence_spans; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.evidence_spans TO polymath;


--
-- Name: SEQUENCE evidence_spans_span_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.evidence_spans_span_id_seq TO polymath;


--
-- Name: TABLE failure_cards; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.failure_cards TO polymath;


--
-- Name: SEQUENCE failure_cards_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.failure_cards_id_seq TO polymath;


--
-- Name: TABLE recipe_cards; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON TABLE public.recipe_cards TO polymath;


--
-- Name: SEQUENCE recipe_cards_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.recipe_cards_id_seq TO polymath;


--
-- PostgreSQL database dump complete
--

\unrestrict Oux20bmvk9Z1U6thHavtXhF6Tvr3VO0ZzmUbYi4jvOQ4Pn673mqKfTtACT94zX7

