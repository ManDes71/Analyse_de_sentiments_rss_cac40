--
-- PostgreSQL database dump
--

-- Dumped from database version 15.13 (Debian 15.13-1.pgdg130+1)
-- Dumped by pg_dump version 16.4

-- Started on 2026-06-20 19:23:48

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
-- TOC entry 223 (class 1259 OID 49620)
-- Name: company_aliases; Type: TABLE; Schema: public; Owner: manu
--

CREATE TABLE public.company_aliases (
    id integer NOT NULL,
    company_id integer NOT NULL,
    alias public.citext NOT NULL,
    alias_norm text
);


ALTER TABLE public.company_aliases OWNER TO manu;

--
-- TOC entry 224 (class 1259 OID 49625)
-- Name: company_aliases_id_seq; Type: SEQUENCE; Schema: public; Owner: manu
--

CREATE SEQUENCE public.company_aliases_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.company_aliases_id_seq OWNER TO manu;

--
-- TOC entry 3554 (class 0 OID 0)
-- Dependencies: 224
-- Name: company_aliases_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: manu
--

ALTER SEQUENCE public.company_aliases_id_seq OWNED BY public.company_aliases.id;


--
-- TOC entry 3394 (class 2604 OID 49650)
-- Name: company_aliases id; Type: DEFAULT; Schema: public; Owner: manu
--

ALTER TABLE ONLY public.company_aliases ALTER COLUMN id SET DEFAULT nextval('public.company_aliases_id_seq'::regclass);


--
-- TOC entry 3547 (class 0 OID 49620)
-- Dependencies: 223
-- Data for Name: company_aliases; Type: TABLE DATA; Schema: public; Owner: manu
--

COPY public.company_aliases (id, company_id, alias, alias_norm) FROM stdin;
10	1	AIR LIQUIDE	air liquide
11	2	AIRBUS	airbus
12	3	ALSTOM	alstom
13	4	ARCURE	arcure
14	5	BNP PARIBAS ACT.A	bnp paribas act.a
15	6	BUREAU VERITAS	bureau veritas
16	7	CARREFOUR	carrefour
17	8	Gaztransport & Technigaz SA	gaztransport & technigaz sa
18	9	NEOEN	neoen
19	10	NEXANS	nexans
20	11	SARTORIUS STED BIO	sartorius sted bio
21	12	SCHNEIDER ELECTRIC	schneider electric
22	13	SOITEC	soitec
23	14	STMICROELECTRONICS	stmicroelectronics
24	15	TOTALENERGIES	totalenergies
25	16	VEOLIA ENVIRON.	veolia environ.
26	17	VERIMATRIX	verimatrix
27	18	SAFRAN	safran
28	19	SANOFI	sanofi
29	20	Delta Plus Group	delta plus group
30	21	CAPGEMINI	capgemini
31	22	Publicis Groupe	publicis groupe
32	23	Roche Bobois S.A.	roche bobois s.a.
33	24	THALES	thales
34	25	Teleperformance SE	teleperformance se
35	26	DASSAULT AVIATION	dassault aviation
36	27	EssilorLuxottica	essilorluxottica
37	28	SAINT-GOBAIN	saint-gobain
38	29	VIRBAC	virbac
39	30	AUBAY	aubay
40	31	EXOSENS	exosens
42	33	AXA	axa
43	34	ENGIE	engie
41	32	Séché Environnement	séché environnement
44	5	BNP PARIBAS	bnp paribas
45	8	GTT	gtt
\.


--
-- TOC entry 3555 (class 0 OID 0)
-- Dependencies: 224
-- Name: company_aliases_id_seq; Type: SEQUENCE SET; Schema: public; Owner: manu
--

SELECT pg_catalog.setval('public.company_aliases_id_seq', 1, false);


--
-- TOC entry 3396 (class 2606 OID 52332)
-- Name: company_aliases company_aliases_company_id_alias_key; Type: CONSTRAINT; Schema: public; Owner: manu
--

ALTER TABLE ONLY public.company_aliases
    ADD CONSTRAINT company_aliases_company_id_alias_key UNIQUE (company_id, alias);


--
-- TOC entry 3398 (class 2606 OID 52334)
-- Name: company_aliases company_aliases_pkey; Type: CONSTRAINT; Schema: public; Owner: manu
--

ALTER TABLE ONLY public.company_aliases
    ADD CONSTRAINT company_aliases_pkey PRIMARY KEY (id);


--
-- TOC entry 3399 (class 1259 OID 52355)
-- Name: idx_company_aliases_company_id; Type: INDEX; Schema: public; Owner: manu
--

CREATE INDEX idx_company_aliases_company_id ON public.company_aliases USING btree (company_id);


--
-- TOC entry 3400 (class 1259 OID 52357)
-- Name: uq_company_alias_norm; Type: INDEX; Schema: public; Owner: manu
--

CREATE UNIQUE INDEX uq_company_alias_norm ON public.company_aliases USING btree (company_id, alias_norm);


--
-- TOC entry 3402 (class 2620 OID 52360)
-- Name: company_aliases trg_company_aliases_norm; Type: TRIGGER; Schema: public; Owner: manu
--

CREATE TRIGGER trg_company_aliases_norm BEFORE INSERT OR UPDATE OF alias ON public.company_aliases FOR EACH ROW EXECUTE FUNCTION public.company_aliases_norm_trg();


--
-- TOC entry 3401 (class 2606 OID 52392)
-- Name: company_aliases company_aliases_company_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: manu
--

ALTER TABLE ONLY public.company_aliases
    ADD CONSTRAINT company_aliases_company_id_fkey FOREIGN KEY (company_id) REFERENCES public.companies(id) ON DELETE CASCADE;


-- Completed on 2026-06-20 19:23:48

--
-- PostgreSQL database dump complete
--

