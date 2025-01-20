-- MySQL dump 10.13  Distrib 9.0.1, for macos14.7 (arm64)
--
-- Host: localhost    Database: spermdetect
-- ------------------------------------------------------
-- Server version	9.0.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Current Database: `spermdetect`
--

CREATE DATABASE /*!32312 IF NOT EXISTS*/ `spermdetect` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;

USE `spermdetect`;

--
-- Table structure for table `admin`
--

DROP TABLE IF EXISTS `admin`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `admin` (
  `aid` int NOT NULL,
  `name` varchar(100) NOT NULL,
  `pass` varchar(100) NOT NULL,
  PRIMARY KEY (`aid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `admin`
--

LOCK TABLES `admin` WRITE;
/*!40000 ALTER TABLE `admin` DISABLE KEYS */;
INSERT INTO `admin` VALUES (1,'admin@gmail.com','admin');
/*!40000 ALTER TABLE `admin` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `patient`
--

DROP TABLE IF EXISTS `patient`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `patient` (
  `uid` int NOT NULL AUTO_INCREMENT,
  `name` varchar(50) NOT NULL,
  `age` int NOT NULL,
  `occupation` varchar(50) NOT NULL,
  `height` float NOT NULL,
  `weight` float NOT NULL,
  `bmi` float NOT NULL,
  `sexual_dysfunction` varchar(20) NOT NULL,
  `alcoholic` varchar(10) NOT NULL,
  `smoker` varchar(10) NOT NULL,
  `drugs` varchar(10) NOT NULL,
  `date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`uid`)
) ENGINE=InnoDB AUTO_INCREMENT=51 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `patient`
--

LOCK TABLES `patient` WRITE;
/*!40000 ALTER TABLE `patient` DISABLE KEYS */;
INSERT INTO `patient` VALUES (50,'kishore',21,'student',1.8,59.8,18.4568,'no','No','No','No','2025-01-17 07:14:46');
/*!40000 ALTER TABLE `patient` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `result`
--

DROP TABLE IF EXISTS `result`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `result` (
  `uid` int NOT NULL,
  `current_sample` int DEFAULT '1',
  `sample1_count` int DEFAULT NULL,
  `sample1_date` timestamp NULL DEFAULT NULL,
  `sample2_count` int DEFAULT NULL,
  `sample2_date` timestamp NULL DEFAULT NULL,
  `sample3_count` int DEFAULT NULL,
  `sample3_date` timestamp NULL DEFAULT NULL,
  `sample4_count` int DEFAULT NULL,
  `sample4_date` timestamp NULL DEFAULT NULL,
  `sample5_count` int DEFAULT NULL,
  `sample5_date` timestamp NULL DEFAULT NULL,
  `sample6_count` int DEFAULT NULL,
  `sample6_date` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`uid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `result`
--

LOCK TABLES `result` WRITE;
/*!40000 ALTER TABLE `result` DISABLE KEYS */;
INSERT INTO `result` VALUES (50,4,195000000,'2025-01-19 11:09:46',361200000,'2025-01-19 11:10:21',369000000,'2025-01-19 11:10:38',NULL,NULL,NULL,NULL,NULL,NULL);
/*!40000 ALTER TABLE `result` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `sample_images`
--

DROP TABLE IF EXISTS `sample_images`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sample_images` (
  `id` int NOT NULL AUTO_INCREMENT,
  `uid` int DEFAULT NULL,
  `sample_number` int DEFAULT NULL,
  `image_number` int DEFAULT NULL,
  `image_path` text,
  `sperm_count` int DEFAULT NULL,
  `date_uploaded` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=867 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `sample_images`
--

LOCK TABLES `sample_images` WRITE;
/*!40000 ALTER TABLE `sample_images` DISABLE KEYS */;
INSERT INTO `sample_images` VALUES (819,50,1,1,'/Users/kishore/Desktop/app/processed_images/sample1_image1.jpg',49,'2025-01-19 11:09:44'),(820,50,1,2,'/Users/kishore/Desktop/app/processed_images/sample1_image2.jpg',135,'2025-01-19 11:09:45'),(821,50,1,3,'/Users/kishore/Desktop/app/processed_images/sample1_image3.jpg',58,'2025-01-19 11:09:45'),(822,50,1,4,'/Users/kishore/Desktop/app/processed_images/sample1_image4.jpg',51,'2025-01-19 11:09:45'),(823,50,1,5,'/Users/kishore/Desktop/app/processed_images/sample1_image5.jpg',47,'2025-01-19 11:09:45'),(824,50,1,6,'/Users/kishore/Desktop/app/processed_images/sample1_image6.jpg',29,'2025-01-19 11:09:45'),(825,50,1,7,'/Users/kishore/Desktop/app/processed_images/sample1_image7.jpg',58,'2025-01-19 11:09:45'),(826,50,1,8,'/Users/kishore/Desktop/app/processed_images/sample1_image8.jpg',58,'2025-01-19 11:09:45'),(827,50,1,9,'/Users/kishore/Desktop/app/processed_images/sample1_image9.jpg',26,'2025-01-19 11:09:45'),(828,50,1,10,'/Users/kishore/Desktop/app/processed_images/sample1_image10.jpg',47,'2025-01-19 11:09:45'),(829,50,1,11,'/Users/kishore/Desktop/app/processed_images/sample1_image11.jpg',90,'2025-01-19 11:09:45'),(830,50,1,12,'/Users/kishore/Desktop/app/processed_images/sample1_image12.jpg',13,'2025-01-19 11:09:45'),(831,50,1,13,'/Users/kishore/Desktop/app/processed_images/sample1_image13.jpg',43,'2025-01-19 11:09:45'),(832,50,1,14,'/Users/kishore/Desktop/app/processed_images/sample1_image14.jpg',123,'2025-01-19 11:09:46'),(833,50,1,15,'/Users/kishore/Desktop/app/processed_images/sample1_image15.jpg',79,'2025-01-19 11:09:46'),(834,50,1,16,'/Users/kishore/Desktop/app/processed_images/sample1_image16.jpg',69,'2025-01-19 11:09:46'),(835,50,2,1,'/Users/kishore/Desktop/app/processed_images/sample2_image1.jpg',132,'2025-01-19 11:10:20'),(836,50,2,2,'/Users/kishore/Desktop/app/processed_images/sample2_image2.jpg',49,'2025-01-19 11:10:20'),(837,50,2,3,'/Users/kishore/Desktop/app/processed_images/sample2_image3.jpg',49,'2025-01-19 11:10:20'),(838,50,2,4,'/Users/kishore/Desktop/app/processed_images/sample2_image4.jpg',119,'2025-01-19 11:10:20'),(839,50,2,5,'/Users/kishore/Desktop/app/processed_images/sample2_image5.jpg',201,'2025-01-19 11:10:20'),(840,50,2,6,'/Users/kishore/Desktop/app/processed_images/sample2_image6.jpg',125,'2025-01-19 11:10:20'),(841,50,2,7,'/Users/kishore/Desktop/app/processed_images/sample2_image7.jpg',92,'2025-01-19 11:10:20'),(842,50,2,8,'/Users/kishore/Desktop/app/processed_images/sample2_image8.jpg',162,'2025-01-19 11:10:21'),(843,50,2,9,'/Users/kishore/Desktop/app/processed_images/sample2_image9.jpg',189,'2025-01-19 11:10:21'),(844,50,2,10,'/Users/kishore/Desktop/app/processed_images/sample2_image10.jpg',68,'2025-01-19 11:10:21'),(845,50,2,11,'/Users/kishore/Desktop/app/processed_images/sample2_image11.jpg',43,'2025-01-19 11:10:21'),(846,50,2,12,'/Users/kishore/Desktop/app/processed_images/sample2_image12.jpg',124,'2025-01-19 11:10:21'),(847,50,2,13,'/Users/kishore/Desktop/app/processed_images/sample2_image13.jpg',81,'2025-01-19 11:10:21'),(848,50,2,14,'/Users/kishore/Desktop/app/processed_images/sample2_image14.jpg',340,'2025-01-19 11:10:21'),(849,50,2,15,'/Users/kishore/Desktop/app/processed_images/sample2_image15.jpg',13,'2025-01-19 11:10:21'),(850,50,2,16,'/Users/kishore/Desktop/app/processed_images/sample2_image16.jpg',19,'2025-01-19 11:10:21'),(851,50,3,1,'/Users/kishore/Desktop/app/processed_images/sample3_image1.jpg',79,'2025-01-19 11:10:37'),(852,50,3,2,'/Users/kishore/Desktop/app/processed_images/sample3_image2.jpg',47,'2025-01-19 11:10:37'),(853,50,3,3,'/Users/kishore/Desktop/app/processed_images/sample3_image3.jpg',216,'2025-01-19 11:10:37'),(854,50,3,4,'/Users/kishore/Desktop/app/processed_images/sample3_image4.jpg',90,'2025-01-19 11:10:37'),(855,50,3,5,'/Users/kishore/Desktop/app/processed_images/sample3_image5.jpg',71,'2025-01-19 11:10:37'),(856,50,3,6,'/Users/kishore/Desktop/app/processed_images/sample3_image6.jpg',57,'2025-01-19 11:10:37'),(857,50,3,7,'/Users/kishore/Desktop/app/processed_images/sample3_image7.jpg',453,'2025-01-19 11:10:37'),(858,50,3,8,'/Users/kishore/Desktop/app/processed_images/sample3_image8.jpg',30,'2025-01-19 11:10:37'),(859,50,3,9,'/Users/kishore/Desktop/app/processed_images/sample3_image9.jpg',161,'2025-01-19 11:10:37'),(860,50,3,10,'/Users/kishore/Desktop/app/processed_images/sample3_image10.jpg',47,'2025-01-19 11:10:38'),(861,50,3,11,'/Users/kishore/Desktop/app/processed_images/sample3_image11.jpg',175,'2025-01-19 11:10:38'),(862,50,3,12,'/Users/kishore/Desktop/app/processed_images/sample3_image12.jpg',59,'2025-01-19 11:10:38'),(863,50,3,13,'/Users/kishore/Desktop/app/processed_images/sample3_image13.jpg',267,'2025-01-19 11:10:38'),(864,50,3,14,'/Users/kishore/Desktop/app/processed_images/sample3_image14.jpg',49,'2025-01-19 11:10:38'),(865,50,3,15,'/Users/kishore/Desktop/app/processed_images/sample3_image15.jpg',25,'2025-01-19 11:10:38'),(866,50,3,16,'/Users/kishore/Desktop/app/processed_images/sample3_image16.jpg',19,'2025-01-19 11:10:38');
/*!40000 ALTER TABLE `sample_images` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `signup`
--

DROP TABLE IF EXISTS `signup`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `signup` (
  `uid` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `email` varchar(100) NOT NULL,
  `pass` varchar(100) NOT NULL,
  PRIMARY KEY (`uid`)
) ENGINE=InnoDB AUTO_INCREMENT=27 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `signup`
--

LOCK TABLES `signup` WRITE;
/*!40000 ALTER TABLE `signup` DISABLE KEYS */;
INSERT INTO `signup` VALUES (1,'123user','123@123','$2b$12$0bN0CXLHZLFeDoVEjDt2veeYpK2bR3lEW8jtEL1uNVZI9IuvGvCdC'),(26,'kishore','kishore@gmail.com','$2b$12$itY63.w9thlMxdVkATevxuoTIjrEImy.XJ36.UgiX7x0zUMz2Dd/2');
/*!40000 ALTER TABLE `signup` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-01-20  9:51:46
