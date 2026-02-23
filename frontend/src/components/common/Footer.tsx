import { Brain, Twitter, Linkedin, Facebook, Heart } from 'lucide-react';
import { Link } from 'react-router-dom';
import './Footer.css';

export const Footer = () => {
    return (
        <footer className="footer">
            <div className="footer-container">
                <div className="footer-grid">
                    {/* Brand Column */}
                    <div className="footer-brand">
                        <div className="footer-logo">
                            <Brain size={32} />
                            <span>Seg-Mind</span>
                        </div>
                        <p className="footer-description">
                            Advanced medical imaging platform utilizing state-of-the-art deep learning
                            for accurate brain tumor segmentation, classification, and growth prediction.
                        </p>
                        <div className="social-links">
                            <a href="#" className="social-icon" aria-label="Twitter">
                                <Twitter size={20} />
                            </a>
                            <a href="#" className="social-icon" aria-label="LinkedIn">
                                <Linkedin size={20} />
                            </a>
                            <a href="#" className="social-icon" aria-label="Facebook">
                                <Facebook size={20} />
                            </a>
                        </div>
                    </div>

                    {/* Quick Links */}
                    <div className="footer-links-group">
                        <h4>Quick Links</h4>
                        <ul>
                            <li><Link to="/">Home</Link></li>
                            <li><a href="#features">Features</a></li>
                            <li><a href="#about">About Us</a></li>
                            <li><a href="#contact">Contact</a></li>
                        </ul>
                    </div>

                    {/* Legal */}
                    <div className="footer-links-group">
                        <h4>Legal</h4>
                        <ul>
                            <li><Link to="/privacy">Privacy Policy</Link></li>
                            <li><Link to="/terms">Terms of Service</Link></li>
                            <li><Link to="/hipaa">HIPAA Compliance</Link></li>
                            <li><Link to="/security">Data Security</Link></li>
                        </ul>
                    </div>

                    {/* Contact Info */}
                    <div className="footer-links-group">
                        <h4>Contact Info</h4>
                        <ul>
                            <li>support@segmind.medical</li>
                            <li>1-800-SEGMIND</li>
                            <li>123 Innovation Drive</li>
                            <li>Medical District, MD 20850</li>
                        </ul>
                    </div>
                </div>

                <div className="footer-bottom">
                    <p className="copyright">
                        &copy; {new Date().getFullYear()} Seg-Mind Medical Systems. All rights reserved.
                    </p>
                    <p className="made-with">
                        Built with <Heart size={14} className="heart-icon" /> for the medical community
                    </p>
                </div>
            </div>
        </footer>
    );
};
