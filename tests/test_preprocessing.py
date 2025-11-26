"""
test_preprocessing.py
Tests unitarios para el módulo de preprocesamiento.
"""

import unittest
import sys
sys.path.append('../src')

from preprocessing import (
    clean_text,
    tokenize_text,
    remove_stopwords,
    apply_stemming,
    apply_lemmatization,
    preprocess_text
)


class TestPreprocessing(unittest.TestCase):
    """Tests para funciones de preprocesamiento de texto"""
    
    def setUp(self):
        """Configuración inicial para los tests"""
        self.sample_text = "Check out this amazing product! https://example.com @user #awesome 123"
        self.simple_text = "This is a simple test"
    
    def test_clean_text_lowercase(self):
        """Test: El texto se convierte a minúsculas"""
        result = clean_text("HELLO WORLD")
        self.assertEqual(result, "hello world")
    
    def test_clean_text_removes_urls(self):
        """Test: Las URLs se eliminan correctamente"""
        text = "Visit https://example.com for more"
        result = clean_text(text)
        self.assertNotIn("https://", result)
        self.assertNotIn("example.com", result)
    
    def test_clean_text_removes_mentions(self):
        """Test: Las menciones (@user) se eliminan"""
        text = "Hello @username how are you"
        result = clean_text(text)
        self.assertNotIn("@username", result)
    
    def test_clean_text_removes_hashtags_symbol(self):
        """Test: El símbolo # se elimina pero la palabra se mantiene"""
        text = "This is #awesome"
        result = clean_text(text)
        self.assertNotIn("#", result)
        self.assertIn("awesome", result)
    
    def test_clean_text_removes_numbers(self):
        """Test: Los números se eliminan"""
        text = "I have 123 apples"
        result = clean_text(text)
        self.assertNotIn("123", result)
    
    def test_clean_text_removes_punctuation(self):
        """Test: La puntuación se elimina"""
        text = "Hello, world! How are you?"
        result = clean_text(text)
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)
        self.assertNotIn("?", result)
    
    def test_tokenize_text(self):
        """Test: El texto se tokeniza correctamente"""
        text = "hello world test"
        tokens = tokenize_text(text)
        self.assertEqual(tokens, ['hello', 'world', 'test'])
    
    def test_tokenize_empty_text(self):
        """Test: Texto vacío retorna lista vacía"""
        tokens = tokenize_text("")
        self.assertEqual(tokens, [])
    
    def test_remove_stopwords(self):
        """Test: Las stopwords se eliminan"""
        tokens = ['this', 'is', 'a', 'test', 'sentence']
        filtered = remove_stopwords(tokens)
        # 'this', 'is', 'a' son stopwords en inglés
        self.assertNotIn('is', filtered)
        self.assertNotIn('a', filtered)
        self.assertIn('test', filtered)
        self.assertIn('sentence', filtered)
    
    def test_apply_stemming(self):
        """Test: El stemming se aplica correctamente"""
        tokens = ['running', 'runs', 'runner']
        stemmed = apply_stemming(tokens)
        # Todos deberían tener la misma raíz 'run'
        self.assertTrue(all('run' in token for token in stemmed))
    
    def test_apply_lemmatization(self):
        """Test: La lematización se aplica correctamente"""
        tokens = ['running', 'runs', 'better']
        lemmatized = apply_lemmatization(tokens)
        # Verificar que se lematizan correctamente
        self.assertIn('running', lemmatized)  # running -> running (verbo)
        self.assertIn('run', lemmatized)      # runs -> run
        self.assertIn('good', lemmatized)     # better -> good
    
    def test_preprocess_text_complete(self):
        """Test: Pipeline completo de preprocesamiento"""
        text = "Hello @user! Check https://example.com #awesome 123"
        result = preprocess_text(text, remove_stops=True, use_stemming=False)
        
        # Verificar que el texto está limpio
        self.assertNotIn("@", result)
        self.assertNotIn("https", result)
        self.assertNotIn("#", result)
        self.assertNotIn("123", result)
    
    def test_preprocess_text_with_stemming(self):
        """Test: Preprocesamiento con stemming"""
        text = "running and jumping are exercises"
        result = preprocess_text(text, use_stemming=True)
        # Debería contener raíces stemmed
        self.assertIn("run", result)
        self.assertIn("jump", result)
    
    def test_preprocess_text_with_lemmatization(self):
        """Test: Preprocesamiento con lematización"""
        text = "The cats are running quickly"
        result = preprocess_text(text, use_lemmatization=True)
        # Las palabras deberían estar lematizadas
        self.assertTrue(len(result) > 0)


class TestPreprocessingEdgeCases(unittest.TestCase):
    """Tests para casos límite"""
    
    def test_clean_text_none_input(self):
        """Test: Input None retorna string vacío"""
        result = clean_text(None)
        self.assertEqual(result, "")
    
    def test_clean_text_empty_string(self):
        """Test: String vacío retorna string vacío"""
        result = clean_text("")
        self.assertEqual(result, "")
    
    def test_tokenize_text_only_stopwords(self):
        """Test: Texto solo con stopwords"""
        tokens = tokenize_text("a an the is")
        self.assertTrue(len(tokens) > 0)


if __name__ == '__main__':
    unittest.main()
