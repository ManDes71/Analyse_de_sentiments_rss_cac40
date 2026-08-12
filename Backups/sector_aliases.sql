--
-- PostgreSQL database dump
--

-- Dumped from database version 15.13 (Debian 15.13-1.pgdg130+1)
-- Dumped by pg_dump version 16.4

-- Started on 2026-06-20 19:24:53

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
-- TOC entry 229 (class 1259 OID 49642)
-- Name: sector_aliases; Type: TABLE; Schema: public; Owner: REMOVED
--

CREATE TABLE public.sector_aliases (
    id integer NOT NULL,
    sector_id integer NOT NULL,
    alias public.citext NOT NULL,
    alias_norm text
);


ALTER TABLE public.sector_aliases OWNER TO REMOVED;

--
-- TOC entry 230 (class 1259 OID 49647)
-- Name: sector_aliases_id_seq; Type: SEQUENCE; Schema: public; Owner: REMOVED
--

CREATE SEQUENCE public.sector_aliases_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.sector_aliases_id_seq OWNER TO REMOVED;

--
-- TOC entry 3554 (class 0 OID 0)
-- Dependencies: 230
-- Name: sector_aliases_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: REMOVED
--

ALTER SEQUENCE public.sector_aliases_id_seq OWNED BY public.sector_aliases.id;


--
-- TOC entry 3394 (class 2604 OID 49652)
-- Name: sector_aliases id; Type: DEFAULT; Schema: public; Owner: REMOVED
--

ALTER TABLE ONLY public.sector_aliases ALTER COLUMN id SET DEFAULT nextval('public.sector_aliases_id_seq'::regclass);


--
-- TOC entry 3547 (class 0 OID 49642)
-- Dependencies: 229
-- Data for Name: sector_aliases; Type: TABLE DATA; Schema: public; Owner: REMOVED
--

COPY public.sector_aliases (id, sector_id, alias, alias_norm) FROM stdin;
1	1	Construction Materials	construction materials
2	1	Matériaux de construction	matériaux de construction
3	1	Paper & Forest	paper & forest
4	1	Papiers & forêts	papiers & forêts
5	1	Steel	steel
6	1	Acier	acier
7	1	Mining	mining
8	1	Mines	mines
9	1	Métaux & mines	métaux & mines
10	1	Chemical industry	chemical industry
11	1	Chimie	chimie
12	1	Matières premières	matières premières
13	1	Basic Materials	basic materials
14	1	Matériaux de base	matériaux de base
15	2	Logistique	logistique
16	2	Transports	transports
17	2	Transport & logistique	transport & logistique
18	2	Machinery	machinery
19	2	Machinerie	machinerie
20	2	Ingénierie	ingénierie
21	2	Aerospace & Defense	aerospace & defense
22	2	Aérospatial & défense	aérospatial & défense
23	2	Aéronautique et défense	aéronautique et défense
24	2	Aéronautique	aéronautique
25	2	Industrials	industrials
26	2	Industrie	industrie
27	3	Intelligence artificielle	intelligence artificielle
28	3	IA	ia
29	3	Cloud	cloud
30	3	Hardware	hardware
31	3	Matériel informatique	matériel informatique
32	3	Software	software
33	3	Logiciels	logiciels
34	3	Semiconductors	semiconductors
35	3	Semi-conducteurs	semi-conducteurs
36	3	Information Technology	information technology
37	3	Technologies de l’information	technologies de l’information
38	3	Tech	tech
39	3	Technologie	technologie
40	4	Bourse	bourse
41	4	Asset Management	asset management
42	4	Gestion d’actifs	gestion d’actifs
43	4	Fintech	fintech
44	4	Assureurs	assureurs
45	4	Assurances	assurances
46	4	Banque	banque
47	4	Banques	banques
48	4	Services financiers diversifiés	services financiers diversifiés
49	4	Finance	finance
50	5	Luxe	luxe
51	5	Grande distribution	grande distribution
52	5	Distribution	distribution
53	5	Retail	retail
54	5	Consumer	consumer
55	5	Biens de consommation	biens de consommation
56	5	Consommation	consommation
57	6	Raffinage	raffinage
58	6	Électricité	électricité
59	6	Énergies renouvelables	énergies renouvelables
60	6	Renouvelables	renouvelables
61	6	Gaz	gaz
62	6	Pétrole	pétrole
63	6	Oil and Gas	oil and gas
64	6	Oil & Gas	oil & gas
65	6	Energie	energie
67	7	MedTech	medtech
68	7	Dispositifs médicaux	dispositifs médicaux
69	7	Biotechnologies	biotechnologies
70	7	Biotech	biotech
71	7	Pharmaceutique	pharmaceutique
72	7	Pharmacie	pharmacie
73	7	Health Care	health care
74	7	Santé	santé
75	8	Waste Management	waste management
76	8	Gestion des déchets	gestion des déchets
77	8	Gaz	gaz
78	8	Électricité	électricité
79	8	Eau	eau
80	8	Utilities	utilities
81	8	Services aux collectivités	services aux collectivités
82	9	Grande distribution alimentaire	grande distribution alimentaire
83	9	Hygiène	hygiène
84	9	Boissons	boissons
85	9	Alimentation	alimentation
86	9	Agroalimentaire	agroalimentaire
87	9	Consumer Staples	consumer staples
88	9	Biens de consommation de base	biens de consommation de base
89	9	Consommation de base	consommation de base
90	10	Streaming	streaming
91	10	Advertising	advertising
92	10	Publicité	publicité
93	10	Médias	médias
94	10	Media	media
95	10	Télécommunications	télécommunications
96	10	Télécoms	télécoms
97	10	Communication Services	communication services
98	10	Services de communication	services de communication
99	11	Distribution spécialisée	distribution spécialisée
100	11	E-commerce	e-commerce
101	11	Hôtellerie	hôtellerie
102	11	Voyage & loisirs	voyage & loisirs
103	11	Auto	auto
104	11	Automobile	automobile
105	11	Consumer Discretionary	consumer discretionary
106	11	Consommation discrétionnaire	consommation discrétionnaire
107	12	Emballage	emballage
108	12	Packaging	packaging
109	12	Construction Materials	construction materials
110	12	Materials	materials
111	12	Matériaux	matériaux
115	13	Assurance	assurance
116	13	Bancaire	bancaire
117	13	Financial Services	financial services
118	13	Services financiers	services financiers
119	14	Service public	service public
120	14	Transport public	transport public
121	14	Collectivités	collectivités
122	14	Utilities	utilities
123	14	Public Services	public services
124	14	Services publics	services publics
\.


--
-- TOC entry 3555 (class 0 OID 0)
-- Dependencies: 230
-- Name: sector_aliases_id_seq; Type: SEQUENCE SET; Schema: public; Owner: REMOVED
--

SELECT pg_catalog.setval('public.sector_aliases_id_seq', 1, false);


--
-- TOC entry 3397 (class 2606 OID 52340)
-- Name: sector_aliases sector_aliases_pkey; Type: CONSTRAINT; Schema: public; Owner: REMOVED
--

ALTER TABLE ONLY public.sector_aliases
    ADD CONSTRAINT sector_aliases_pkey PRIMARY KEY (id);


--
-- TOC entry 3399 (class 2606 OID 52342)
-- Name: sector_aliases sector_aliases_sector_id_alias_uk; Type: CONSTRAINT; Schema: public; Owner: REMOVED
--

ALTER TABLE ONLY public.sector_aliases
    ADD CONSTRAINT sector_aliases_sector_id_alias_uk UNIQUE (sector_id, alias);


--
-- TOC entry 3395 (class 1259 OID 52356)
-- Name: idx_sector_aliases_sector_id; Type: INDEX; Schema: public; Owner: REMOVED
--

CREATE INDEX idx_sector_aliases_sector_id ON public.sector_aliases USING btree (sector_id);


--
-- TOC entry 3400 (class 1259 OID 52358)
-- Name: uq_sector_alias_norm; Type: INDEX; Schema: public; Owner: REMOVED
--

CREATE UNIQUE INDEX uq_sector_alias_norm ON public.sector_aliases USING btree (sector_id, alias_norm);


--
-- TOC entry 3402 (class 2620 OID 52361)
-- Name: sector_aliases trg_sector_aliases_norm; Type: TRIGGER; Schema: public; Owner: REMOVED
--

CREATE TRIGGER trg_sector_aliases_norm BEFORE INSERT OR UPDATE OF alias ON public.sector_aliases FOR EACH ROW EXECUTE FUNCTION public.sector_aliases_norm_trg();


--
-- TOC entry 3401 (class 2606 OID 52397)
-- Name: sector_aliases sector_aliases_sector_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: REMOVED
--

ALTER TABLE ONLY public.sector_aliases
    ADD CONSTRAINT sector_aliases_sector_id_fkey FOREIGN KEY (sector_id) REFERENCES public.sectors(id) ON DELETE CASCADE;


-- Completed on 2026-06-20 19:24:53

--
-- PostgreSQL database dump complete
--

