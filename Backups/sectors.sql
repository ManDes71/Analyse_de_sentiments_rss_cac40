--
-- PostgreSQL database dump
--

-- Dumped from database version 15.13 (Debian 15.13-1.pgdg130+1)
-- Dumped by pg_dump version 16.4

-- Started on 2026-06-20 19:25:23

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
-- TOC entry 227 (class 1259 OID 49633)
-- Name: sectors; Type: TABLE; Schema: public; Owner: manu
--

CREATE TABLE public.sectors (
    id integer NOT NULL,
    name text NOT NULL
);


ALTER TABLE public.sectors OWNER TO manu;

--
-- TOC entry 231 (class 1259 OID 49648)
-- Name: sectors_id_seq; Type: SEQUENCE; Schema: public; Owner: manu
--

CREATE SEQUENCE public.sectors_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.sectors_id_seq OWNER TO manu;

--
-- TOC entry 3550 (class 0 OID 0)
-- Dependencies: 231
-- Name: sectors_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: manu
--

ALTER SEQUENCE public.sectors_id_seq OWNED BY public.sectors.id;


--
-- TOC entry 3394 (class 2604 OID 49653)
-- Name: sectors id; Type: DEFAULT; Schema: public; Owner: manu
--

ALTER TABLE ONLY public.sectors ALTER COLUMN id SET DEFAULT nextval('public.sectors_id_seq'::regclass);


--
-- TOC entry 3543 (class 0 OID 49633)
-- Dependencies: 227
-- Data for Name: sectors; Type: TABLE DATA; Schema: public; Owner: manu
--

COPY public.sectors (id, name) FROM stdin;
1	Matériaux de base
2	Industrie
3	Technologie
4	Finance
5	Consommation
6	Énergie
7	Santé
8	Services aux collectivités
9	Consommation de base
10	Services de communication
11	Consommation discrétionnaire
12	Matériaux
13	Services financiers
14	Services publics
\.


--
-- TOC entry 3551 (class 0 OID 0)
-- Dependencies: 231
-- Name: sectors_id_seq; Type: SEQUENCE SET; Schema: public; Owner: manu
--

SELECT pg_catalog.setval('public.sectors_id_seq', 1, false);


--
-- TOC entry 3396 (class 2606 OID 52344)
-- Name: sectors sectors_name_key; Type: CONSTRAINT; Schema: public; Owner: manu
--

ALTER TABLE ONLY public.sectors
    ADD CONSTRAINT sectors_name_key UNIQUE (name);


--
-- TOC entry 3398 (class 2606 OID 52346)
-- Name: sectors sectors_pkey; Type: CONSTRAINT; Schema: public; Owner: manu
--

ALTER TABLE ONLY public.sectors
    ADD CONSTRAINT sectors_pkey PRIMARY KEY (id);


-- Completed on 2026-06-20 19:25:24

--
-- PostgreSQL database dump complete
--

