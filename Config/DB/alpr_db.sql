-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: localhost:8889
-- Generation Time: May 19, 2025 at 12:16 PM
-- Server version: 8.0.40
-- PHP Version: 8.3.14

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `alpr_db`
--

-- --------------------------------------------------------

--
-- Table structure for table `detections`
--

CREATE TABLE `detections` (
  `id` varchar(36) NOT NULL,
  `plate_number` varchar(20) NOT NULL,
  `image_path` varchar(255) NOT NULL,
  `type` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `color` varchar(20) NOT NULL,
  `timestamp` datetime DEFAULT NULL,
  `is_validated` tinyint(1) DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `detections`
--

INSERT INTO `detections` (`id`, `plate_number`, `image_path`, `type`, `color`, `timestamp`, `is_validated`) VALUES
('0100f0bf-2879-4707-b63d-384eed0979c5', 'KORLANTASPOLRI', 'Storage/Uploads/gambar9.jpeg', 'CIVIL', 'UNKNOWN', '2025-05-19 01:28:44', 0),
('2f70df34-83cf-4ca0-81f9-0b9171d4cb61', 'B2156TOR', 'static/uploads/gambar2.jpeg', 'CIVIL', 'WHITE', '2025-05-18 15:43:11', 0),
('6703dffa-876f-436b-bbf3-e4e94b6558b6', 'B1067NBF', 'static/uploads/gambar5.jpeg', 'CIVIL', 'WHITE', '2025-05-18 15:43:01', 0),
('88e14ad9-4f08-4111-9c49-8560d0554268', '311000', 'static/uploads/gambar3.jpeg', 'CIVIL', 'RED', '2025-05-18 15:49:46', 0),
('a2a034cd-4eeb-4188-a43e-fdedd350b75b', '311000', 'static/uploads/gambar3.jpeg', 'CIVIL', 'YELLOW', '2025-05-18 15:46:39', 0),
('a302862e-594a-4dbf-9673-acd38eb0dc14', '204945', 'static/uploads/gambar8.jpeg', 'CIVIL', 'GREEN', '2025-05-18 15:44:45', 0),
('a80022e2-465c-4681-892b-325e34254fe0', 'B2156TOR', 'static/uploads/gambar4.jpeg', 'CIVIL', 'UNKNOWN', '2025-05-18 15:49:43', 0),
('a9928689-3716-4ef2-9d9a-910667401fcf', '204945', 'static/uploads/gambar8.jpeg', 'CIVIL', 'BLACK', '2025-05-18 15:44:33', 0),
('b998900a-3ea4-43f0-a028-af1fbd42c9f8', 'B1001ZZZ', 'static/uploads/gambar1.jpeg', 'CIVIL', 'WHITE', '2025-05-18 15:43:18', 0),
('bdc71d94-04a5-4a83-8b62-4126e85b1a57', 'N4560R', 'static/uploads/gambar7.jpeg', 'CIVIL', 'BLACK', '2025-05-18 15:51:30', 0),
('d9704642-7699-42a0-96fb-6ee47c347844', 'B1067NBF', 'static/uploads/gambar5.jpeg', 'CIVIL', 'BLACK', '2025-05-18 15:46:54', 0),
('dcb8da13-b125-496f-a829-20de29fd19ca', 'BH1512TT', 'static/uploads/gambar6.jpeg', 'CIVIL', 'UNKNOWN', '2025-05-18 15:46:18', 0),
('df998a85-b997-4401-9ddf-c02a0159e628', '311000', 'static/uploads/gambar3.jpeg', 'CIVIL', 'GREEN', '2025-05-18 15:43:14', 0),
('e07ec271-7047-4942-ad1f-9d83b781ae60', 'B1067NBF', 'static/uploads/gambar5.jpeg', 'CIVIL', 'UNKNOWN', '2025-05-18 15:46:05', 0),
('e5e5be87-0ef1-4caa-b096-331fc0b6f2e4', 'OM3845AR', 'Storage/Uploads/gambar10.jpeg', 'CIVIL', 'UNKNOWN', '2025-05-19 01:29:20', 0),
('e6d6f14e-482d-4484-8626-92783ccd8a43', 'BH1512TT', 'static/uploads/gambar6.jpeg', 'CIVIL', 'GREEN', '2025-05-18 15:43:31', 0),
('efbbc96f-a919-42dd-b9ea-834ce24aa150', 'BH1512TT', 'static/uploads/gambar6.jpeg', 'CIVIL', 'BLACK', '2025-05-18 15:43:55', 0),
('f199d8f3-9e14-46c1-ba55-68b4407c9738', '311000', 'static/uploads/gambar3.jpeg', 'CIVIL', 'UNKNOWN', '2025-05-18 15:51:07', 0);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `detections`
--
ALTER TABLE `detections`
  ADD PRIMARY KEY (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
