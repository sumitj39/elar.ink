--
-- PostgreSQL database dump
--

-- Dumped from database version 14.9 (Ubuntu 14.9-1.pgdg22.04+1)
-- Dumped by pg_dump version 14.9 (Ubuntu 14.9-1.pgdg22.04+1)

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
-- Name: pgcrypto; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;


--
-- Name: EXTENSION pgcrypto; Type: COMMENT; Schema: -; Owner:
--

COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner:
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat access method';


--
-- Name: entry_status; Type: TYPE; Schema: public; Owner: alaradmin
--

CREATE TYPE public.entry_status AS ENUM (
    'pending',
    'enabled',
    'disabled'
);


ALTER TYPE public.entry_status OWNER TO alaradmin;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: comments; Type: TABLE; Schema: public; Owner: alaradmin
--

CREATE TABLE public.comments (
    id integer NOT NULL,
    from_id integer NOT NULL,
    to_id integer,
    comments text DEFAULT ''::text NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.comments OWNER TO alaradmin;

--
-- Name: comments_id_seq; Type: SEQUENCE; Schema: public; Owner: alaradmin
--

CREATE SEQUENCE public.comments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.comments_id_seq OWNER TO alaradmin;

--
-- Name: comments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: alaradmin
--

ALTER SEQUENCE public.comments_id_seq OWNED BY public.comments.id;


--
-- Name: embeddings; Type: TABLE; Schema: public; Owner: alaradmin
--

CREATE TABLE public.embeddings (
    id bigint NOT NULL,
    entry_id bigint,
    embedding public.vector(384)
);


ALTER TABLE public.embeddings OWNER TO alaradmin;

--
-- Name: embeddings_id_seq; Type: SEQUENCE; Schema: public; Owner: alaradmin
--

CREATE SEQUENCE public.embeddings_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.embeddings_id_seq OWNER TO alaradmin;

--
-- Name: embeddings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: alaradmin
--

ALTER SEQUENCE public.embeddings_id_seq OWNED BY public.embeddings.id;


--
-- Name: entries; Type: TABLE; Schema: public; Owner: alaradmin
--

CREATE TABLE public.entries (
    id integer NOT NULL,
    guid uuid DEFAULT gen_random_uuid() NOT NULL,
    content text NOT NULL,
    initial text DEFAULT ''::text NOT NULL,
    weight numeric DEFAULT 0 NOT NULL,
    tokens tsvector DEFAULT ''::tsvector NOT NULL,
    lang text NOT NULL,
    tags text[] DEFAULT '{}'::text[] NOT NULL,
    phones text[] DEFAULT '{}'::text[] NOT NULL,
    notes text DEFAULT ''::text NOT NULL,
    status public.entry_status DEFAULT 'enabled'::public.entry_status NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT entries_content_check CHECK ((content <> ''::text)),
    CONSTRAINT entries_lang_check CHECK ((lang <> ''::text))
);


ALTER TABLE public.entries OWNER TO alaradmin;

--
-- Name: entries_id_seq; Type: SEQUENCE; Schema: public; Owner: alaradmin
--

CREATE SEQUENCE public.entries_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.entries_id_seq OWNER TO alaradmin;

--
-- Name: entries_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: alaradmin
--

ALTER SEQUENCE public.entries_id_seq OWNED BY public.entries.id;


--
-- Name: relations; Type: TABLE; Schema: public; Owner: alaradmin
--

CREATE TABLE public.relations (
    id integer NOT NULL,
    from_id integer,
    to_id integer,
    types text[] DEFAULT '{}'::text[] NOT NULL,
    tags text[] DEFAULT '{}'::text[] NOT NULL,
    notes text DEFAULT ''::text NOT NULL,
    weight numeric DEFAULT 0,
    status public.entry_status DEFAULT 'enabled'::public.entry_status NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.relations OWNER TO alaradmin;

--
-- Name: relations_id_seq; Type: SEQUENCE; Schema: public; Owner: alaradmin
--

CREATE SEQUENCE public.relations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.relations_id_seq OWNER TO alaradmin;

--
-- Name: relations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: alaradmin
--

ALTER SEQUENCE public.relations_id_seq OWNED BY public.relations.id;


--
-- Name: comments id; Type: DEFAULT; Schema: public; Owner: alaradmin
--

ALTER TABLE ONLY public.comments ALTER COLUMN id SET DEFAULT nextval('public.comments_id_seq'::regclass);


--
-- Name: embeddings id; Type: DEFAULT; Schema: public; Owner: alaradmin
--

ALTER TABLE ONLY public.embeddings ALTER COLUMN id SET DEFAULT nextval('public.embeddings_id_seq'::regclass);


--
-- Name: entries id; Type: DEFAULT; Schema: public; Owner: alaradmin
--

ALTER TABLE ONLY public.entries ALTER COLUMN id SET DEFAULT nextval('public.entries_id_seq'::regclass);


--
-- Name: relations id; Type: DEFAULT; Schema: public; Owner: alaradmin
--

ALTER TABLE ONLY public.relations ALTER COLUMN id SET DEFAULT nextval('public.relations_id_seq'::regclass);


--
-- Name: comments comments_pkey; Type: CONSTRAINT; Schema: public; Owner: alaradmin
--

ALTER TABLE ONLY public.comments
    ADD CONSTRAINT comments_pkey PRIMARY KEY (id);


--
-- Name: embeddings embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: alaradmin
--

ALTER TABLE ONLY public.embeddings
    ADD CONSTRAINT embeddings_pkey PRIMARY KEY (id);


--
-- Name: entries entries_guid_key; Type: CONSTRAINT; Schema: public; Owner: alaradmin
--

ALTER TABLE ONLY public.entries
    ADD CONSTRAINT entries_guid_key UNIQUE (guid);


--
-- Name: entries entries_pkey; Type: CONSTRAINT; Schema: public; Owner: alaradmin
--

ALTER TABLE ONLY public.entries
    ADD CONSTRAINT entries_pkey PRIMARY KEY (id);


--
-- Name: relations relations_pkey; Type: CONSTRAINT; Schema: public; Owner: alaradmin
--

ALTER TABLE ONLY public.relations
    ADD CONSTRAINT relations_pkey PRIMARY KEY (id);


--
-- Name: embeddings_embedding_idx; Type: INDEX; Schema: public; Owner: alaradmin
--

CREATE INDEX embeddings_embedding_idx ON public.embeddings USING hnsw (embedding public.vector_cosine_ops);


--
-- Name: idx_entries_content; Type: INDEX; Schema: public; Owner: alaradmin
--

CREATE INDEX idx_entries_content ON public.entries USING btree (lower("substring"(content, 0, 50)));


--
-- Name: idx_entries_initial; Type: INDEX; Schema: public; Owner: alaradmin
--

CREATE INDEX idx_entries_initial ON public.entries USING btree (initial);


--
-- Name: idx_entries_lang; Type: INDEX; Schema: public; Owner: alaradmin
--

CREATE INDEX idx_entries_lang ON public.entries USING btree (lang);


--
-- Name: idx_entries_tags; Type: INDEX; Schema: public; Owner: alaradmin
--

CREATE INDEX idx_entries_tags ON public.entries USING btree (tags);


--
-- Name: idx_entries_tokens; Type: INDEX; Schema: public; Owner: alaradmin
--

CREATE INDEX idx_entries_tokens ON public.entries USING gin (tokens);


--
-- Name: idx_relations; Type: INDEX; Schema: public; Owner: alaradmin
--

CREATE UNIQUE INDEX idx_relations ON public.relations USING btree (from_id, to_id);


--
-- Name: relations_to_id_idx; Type: INDEX; Schema: public; Owner: alaradmin
--

CREATE INDEX relations_to_id_idx ON public.relations USING btree (to_id);


--
-- Name: comments comments_from_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: alaradmin
--

ALTER TABLE ONLY public.comments
    ADD CONSTRAINT comments_from_id_fkey FOREIGN KEY (from_id) REFERENCES public.entries(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: comments comments_to_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: alaradmin
--

ALTER TABLE ONLY public.comments
    ADD CONSTRAINT comments_to_id_fkey FOREIGN KEY (to_id) REFERENCES public.entries(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: relations relations_from_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: alaradmin
--

ALTER TABLE ONLY public.relations
    ADD CONSTRAINT relations_from_id_fkey FOREIGN KEY (from_id) REFERENCES public.entries(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: relations relations_to_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: alaradmin
--

ALTER TABLE ONLY public.relations
    ADD CONSTRAINT relations_to_id_fkey FOREIGN KEY (to_id) REFERENCES public.entries(id) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

