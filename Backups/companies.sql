--
-- PostgreSQL database dump
--

-- Dumped from database version 15.13 (Debian 15.13-1.pgdg130+1)
-- Dumped by pg_dump version 16.4

-- Started on 2026-06-20 19:23:12

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
-- TOC entry 220 (class 1259 OID 49609)
-- Name: companies; Type: TABLE; Schema: public; Owner: manu
--

CREATE TABLE public.companies (
    id integer NOT NULL,
    name text NOT NULL,
    isin text,
    ticker text,
    country text,
    sector_id integer
);


ALTER TABLE public.companies OWNER TO manu;

--
-- TOC entry 222 (class 1259 OID 49619)
-- Name: companies_id_seq; Type: SEQUENCE; Schema: public; Owner: manu
--

CREATE SEQUENCE public.companies_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.companies_id_seq OWNER TO manu;

--
-- TOC entry 3553 (class 0 OID 0)
-- Dependencies: 222
-- Name: companies_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: manu
--

ALTER SEQUENCE public.companies_id_seq OWNED BY public.companies.id;


--
-- TOC entry 3394 (class 2604 OID 49649)
-- Name: companies id; Type: DEFAULT; Schema: public; Owner: manu
--

ALTER TABLE ONLY public.companies ALTER COLUMN id SET DEFAULT nextval('public.companies_id_seq'::regclass);


--
-- TOC entry 3546 (class 0 OID 49609)
-- Dependencies: 220
-- Data for Name: companies; Type: TABLE DATA; Schema: public; Owner: manu
--

COPY public.companies (id, name, isin, ticker, country, sector_id) FROM stdin;
1	AIR LIQUIDE	FR0000120073	AI.PA	France	1
2	AIRBUS	NL0000235190	AIR.PA	France	2
3	ALSTOM	FR0010220475	ALO.PA	France	2
4	ARCURE	FR0013398997	ALCUR.PA	France	3
5	BNP PARIBAS ACT.A	FR0000131104	BNP.PA	France	4
6	BUREAU VERITAS	FR0006174348	BVI.PA	France	2
7	CARREFOUR	FR0000120172	CA.PA	France	5
8	Gaztransport & Technigaz SA	FR0011726835	GTT.PA	France	6
9	NEOEN	FR0011675362	NEOEN.PA	France	6
10	NEXANS	FR0000044448	NEX.PA	France	2
11	SARTORIUS STED BIO	FR0013154002	DIM.PA	France	7
12	SCHNEIDER ELECTRIC	FR0000121972	SU.PA	France	2
13	SOITEC	FR0013227113	SOI.PA	France	3
14	STMICROELECTRONICS	NL0000226223	STMPA.PA	France	3
15	TOTALENERGIES	FR0000120271	TTE.PA	France	6
16	VEOLIA ENVIRON.	FR0000124141	VIE.PA	France	8
17	VERIMATRIX	FR0010291245	VMX.PA	France	3
18	SAFRAN	FR0000073272	SAF.PA	France	3
19	SANOFI	FR0000120578	SAN.PA	France	7
20	Delta Plus Group	FR0013283108	ALDLT.PA	France	9
21	CAPGEMINI	FR0000125338	CAP.PA	France	3
22	Publicis Groupe	FR0000130577	PUB.PA	France	10
23	Roche Bobois S.A.	FR0013344173	RBO.PA	France	11
24	THALES	FR0000121329	HO.PA	France	3
25	Teleperformance SE	FR0000051807	TEP.PA	France	2
26	DASSAULT AVIATION	FR0014004L86	AM.PA	France	2
27	EssilorLuxottica	FR0000121667	EL.PA	France	7
28	SAINT-GOBAIN	FR0000125007	SGO.PA	France	12
29	VIRBAC	FR0000031577	VIRP.PA	France	7
30	AUBAY	FR0000063737	AUB.PA	France	3
31	EXOSENS	FR001400Q9V2	EXENS.PA	France	2
32	Séché Environnement	FR0000039109	SCHP.PA	France	2
33	AXA	FR0000120628	CS.PA	France	13
34	ENGIE	FR0010208488	ENGI.PA	France	14
\.


--
-- TOC entry 3554 (class 0 OID 0)
-- Dependencies: 222
-- Name: companies_id_seq; Type: SEQUENCE SET; Schema: public; Owner: manu
--

SELECT pg_catalog.setval('public.companies_id_seq', 1, false);


--
-- TOC entry 3396 (class 2606 OID 52326)
-- Name: companies companies_isin_key; Type: CONSTRAINT; Schema: public; Owner: manu
--

ALTER TABLE ONLY public.companies
    ADD CONSTRAINT companies_isin_key UNIQUE (isin);


--
-- TOC entry 3398 (class 2606 OID 52328)
-- Name: companies companies_name_key; Type: CONSTRAINT; Schema: public; Owner: manu
--

ALTER TABLE ONLY public.companies
    ADD CONSTRAINT companies_name_key UNIQUE (name);


--
-- TOC entry 3400 (class 2606 OID 52330)
-- Name: companies companies_pkey; Type: CONSTRAINT; Schema: public; Owner: manu
--

ALTER TABLE ONLY public.companies
    ADD CONSTRAINT companies_pkey PRIMARY KEY (id);


--
-- TOC entry 3401 (class 2606 OID 52387)
-- Name: companies companies_sector_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: manu
--

ALTER TABLE ONLY public.companies
    ADD CONSTRAINT companies_sector_id_fkey FOREIGN KEY (sector_id) REFERENCES public.sectors(id) ON DELETE SET NULL;


-- Completed on 2026-06-20 19:23:12

--
-- PostgreSQL database dump complete
--

