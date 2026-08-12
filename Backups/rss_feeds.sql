--
-- PostgreSQL database dump
--

-- Dumped from database version 15.13 (Debian 15.13-1.pgdg130+1)
-- Dumped by pg_dump version 16.4

-- Started on 2026-06-20 19:24:21

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

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 225 (class 1259 OID 49626)
-- Name: rss_feeds; Type: TABLE; Schema: public; Owner: REMOVED
--

CREATE TABLE public.rss_feeds (
    id integer NOT NULL,
    name text NOT NULL,
    url text NOT NULL,
    source_domain text NOT NULL,
    enabled boolean DEFAULT true NOT NULL,
    etag text,
    last_modified text,
    last_fetched_at timestamp with time zone
);


ALTER TABLE public.rss_feeds OWNER TO REMOVED;

--
-- TOC entry 226 (class 1259 OID 49632)
-- Name: rss_feeds_id_seq; Type: SEQUENCE; Schema: public; Owner: REMOVED
--

CREATE SEQUENCE public.rss_feeds_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.rss_feeds_id_seq OWNER TO REMOVED;

--
-- TOC entry 3551 (class 0 OID 0)
-- Dependencies: 226
-- Name: rss_feeds_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: REMOVED
--

ALTER SEQUENCE public.rss_feeds_id_seq OWNED BY public.rss_feeds.id;


--
-- TOC entry 3394 (class 2604 OID 49651)
-- Name: rss_feeds id; Type: DEFAULT; Schema: public; Owner: REMOVED
--

ALTER TABLE ONLY public.rss_feeds ALTER COLUMN id SET DEFAULT nextval('public.rss_feeds_id_seq'::regclass);


--
-- TOC entry 3544 (class 0 OID 49626)
-- Dependencies: 225
-- Data for Name: rss_feeds; Type: TABLE DATA; Schema: public; Owner: REMOVED
--

COPY public.rss_feeds (id, name, url, source_domain, enabled, etag, last_modified, last_fetched_at) FROM stdin;
1	Investir - Conseils	https://services.lesechos.fr/rss/investir-conseils-boursiers.xml	services.lesechos.fr	f	\N	\N	\N
2	Investir - Actus valeurs	https://services.lesechos.fr/rss/investir-actualites-valeurs.xml	services.lesechos.fr	f	\N	\N	\N
3	investir-marches-indices	https://services.lesechos.fr/rss/investir-marches-indices.xml	services.lesechos.fr	f	\N	\N	\N
4	les-echos-finance-marches	https://services.lesechos.fr/rss/les-echos-finance-marches.xml	services.lesechos.fr	f	\N	\N	\N
5	les-echos-tech-medias	https://services.lesechos.fr/rss/les-echos-tech-medias.xml	services.lesechos.fr	f	\N	\N	\N
6	abcbourse-lastAnalysis	https://www.abcbourse.com/rss/lastAnalysisRSS	abcbourse.com	t	\N	\N	\N
7	abcbourse-displaynews	https://www.abcbourse.com/rss/displaynewsrss	abcbourse.com	t	\N	\N	\N
8	abcbourse-chroniques	https://www.abcbourse.com/rss/chroniquesrss	abcbourse.com	t	\N	\N	\N
9	cerclefinance-rss	http://www.cerclefinance.com/rss/rss.asp	cerclefinance.com	t	\N	\N	\N
10	easybourse-media	https://www.easybourse.com/feeds/media/	easybourse.com	t	\N	\N	\N
11	easybourse-news	https://www.easybourse.com/feeds/news/fr/	easybourse.com	t	\N	\N	\N
12	tradingsat-bourse	https://www.tradingsat.com/rssbourse.xml	tradingsat.com	t	\N	\N	\N
13	bfmtv-crypto	https://www.bfmtv.com/rss/crypto/	bfmtv.com	t	\N	\N	\N
14	investing-news	https://fr.investing.com/rss/news.rss	investing.com	t	\N	\N	\N
\.


--
-- TOC entry 3552 (class 0 OID 0)
-- Dependencies: 226
-- Name: rss_feeds_id_seq; Type: SEQUENCE SET; Schema: public; Owner: REMOVED
--

SELECT pg_catalog.setval('public.rss_feeds_id_seq', 1, false);


--
-- TOC entry 3397 (class 2606 OID 52336)
-- Name: rss_feeds rss_feeds_pkey; Type: CONSTRAINT; Schema: public; Owner: REMOVED
--

ALTER TABLE ONLY public.rss_feeds
    ADD CONSTRAINT rss_feeds_pkey PRIMARY KEY (id);


--
-- TOC entry 3399 (class 2606 OID 52338)
-- Name: rss_feeds rss_feeds_url_key; Type: CONSTRAINT; Schema: public; Owner: REMOVED
--

ALTER TABLE ONLY public.rss_feeds
    ADD CONSTRAINT rss_feeds_url_key UNIQUE (url);


-- Completed on 2026-06-20 19:24:21

--
-- PostgreSQL database dump complete
--

